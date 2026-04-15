import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
import argparse
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, StableDiffusionControlNetPipeline, UNet2DConditionModel, UniPCMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb
class MoEUNetWrapper(nn.Module):
    def __init__(self, original_unet, num_classes=5):
        super().__init__()
        self.unet = original_unet
        self.num_classes = num_classes

        old_conv = self.unet.conv_out
        in_c = old_conv.in_channels
        out_c = old_conv.out_channels

        self.experts = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            for _ in range(num_classes)
        ])

        for expert in self.experts:
            expert.weight.data.copy_(old_conv.weight.data)
            if old_conv.bias is not None:
                expert.bias.data.copy_(old_conv.bias.data)

        self.unet.conv_out = nn.Identity()

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        class_id=None,
        cross_attention_kwargs=None,
        return_dict=True,
        **kwargs
    ):
        if class_id is None and cross_attention_kwargs is not None:
            if "class_id" in cross_attention_kwargs:
                class_id = cross_attention_kwargs.pop("class_id")

        outputs = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=True,
            **kwargs
        )

        features = outputs.sample

        if class_id is None:
            raise ValueError("MoE mode must pass in class_id.")

        if not torch.is_tensor(class_id):
            class_id = torch.tensor(class_id, device=features.device)

        class_id = class_id.to(device=features.device, dtype=torch.long).view(-1)

        if class_id.shape[0] != features.shape[0]:
            raise ValueError(
                f"class_id batch size ({class_id.shape[0]}) does not match features batch size ({features.shape[0]})."
            )

        out = torch.empty(
            features.shape[0],
            self.experts[0].out_channels,
            features.shape[2],
            features.shape[3],
            device=features.device,
            dtype=features.dtype,
        )

        unique_cids = torch.unique(class_id).tolist()
        for cid in unique_cids:
            cid = int(cid)
            if cid < 0 or cid >= self.num_classes:
                raise ValueError(f"class_id={cid} out of range, should be in [0, {self.num_classes - 1}].")

            mask = (class_id == cid)
            out[mask] = self.experts[cid](features[mask])

        if return_dict:
            return UNet2DConditionOutput(sample=out)
        return (out,)

    @property
    def config(self):
        return self.unet.config

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
}
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


def log_validation(
    pipeline,
    args,
    accelerator,
    generator,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    original_image = download_image(args.val_image_url)
    edited_images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        for _ in range(args.num_validation_images):
            edited_images.append(
                pipeline(
                    args.validation_prompt,
                    image=original_image,
                    num_inference_steps=20,
                    image_guidance_scale=1.5,
                    guidance_scale=7,
                    generator=generator,
                    cross_attention_kwargs={
                        "class_id": torch.tensor([args.validation_class_id], device=accelerator.device, dtype=torch.long)
                    },
                ).images[0]
            )

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
            for edited_image in edited_images:
                wandb_table.add_data(wandb.Image(original_image), wandb.Image(edited_image), args.validation_prompt)
            tracker.log({"validation": wandb_table})


def parse_args():
    parser = argparse.ArgumentParser(description="SG-Diff MoE & ControlNet joint training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
    "--class_json",
    type=str,
    default=None,
    help="The class information output from the previous stage. It can be a summary json file or a directory of json files."
    )

    parser.add_argument(
    "--num_experts",
    type=int,
    default=5,
    help="Number of experts, default 5 classes."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model. If not specified, unet is loaded from `--pretrained_model_name_or_path`.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--prior_dir",
        type=str,
        default=None,
        help="Path to the extracted 6-channel .npy probability maps for prior injection.",
    )
    parser.add_argument(
        "--sdf_dir",
        type=str,
        default=None,
        help="Path to the extracted 1-channel .npy SDF maps for ControlNet conditioning.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Whether to freeze the U-Net backbone (except conv_in) for stage 1 warmup training.",
    )
    parser.add_argument(
        "--original_image_column",
        type=str,
        default="input-image",
        help="The column of the dataset containing the original image on which edits where made.",
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default="edited_image",
        help="The column of the dataset containing the edited image.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default="edit_prompt",
        help="The column of the dataset containing the edit instruction.",
    )
    parser.add_argument(
        "--val_image_url",
        type=str,
        default=None,
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--validation_class_id",
        type=int,
        default=0,
        help="Category IDs used for validation/debug inference.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sgdiff-moe-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training SG-Diff. See section 3.2.1 in the paper: https://huggingface.co/papers/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args
def build_class_id_getter(class_json_path):
    """
    Support two forms:
    1) class_json_path is a summary json file
       For example:
       {
         "xxx.png": 0,
         "yyy.png": 3
       }
       or
       {
         "xxx": {"class_id": 0},
         "yyy": {"class_id": 3}
       }

    2) class_json_path is a directory
       Each image in the directory corresponds to a json, for example, xxx.json contains {"class_id": 2}
    """
    if class_json_path is None:
        raise ValueError("Task 5 requires class_id routing, so --class_json cannot be empty.")

    p = Path(class_json_path)

    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        def _getter(image_name):
            name = Path(image_name).name
            stem = Path(image_name).stem

            candidates = [name, stem]

            for k in candidates:
                if k in mapping:
                    v = mapping[k]
                    if isinstance(v, dict):
                        if "class_id" in v:
                            return int(v["class_id"])
                        if "pred_class" in v:
                            return int(v["pred_class"])
                        if "cls" in v:
                            return int(v["cls"])
                    return int(v)

            raise KeyError(f"Image {image_name} class_id not found in class_json mapping.")

        return _getter

    elif p.is_dir():
        def _getter(image_name):
            base_name = Path(image_name).name
            stem = Path(image_name).stem
            prefix = base_name.split("_")[0]

            candidates = [
                p / f"{stem}.json",
                p / f"{stem}_cls.json",
                p / f"{prefix}.json",
                p / f"{prefix}_cls.json",
            ]

            json_file = None
            for cand in candidates:
                if cand.exists():
                    json_file = cand
                    break

            if json_file is None:
                raise FileNotFoundError(
                 f"Image {image_name} class json not found. Tried: {[str(x) for x in candidates]}"
             )

            with open(json_file, "r", encoding="utf-8") as f:
                info = json.load(f)

            if "class_id" in info:
                return int(info["class_id"])
            if "predicted_class" in info:
                return int(info["predicted_class"])
            if "pred_class" in info:
                return int(info["pred_class"])
            if "cls" in info:
                return int(info["cls"])
            if "morphology_label" in info:
                return int(info["morphology_label"])

            raise KeyError(
                f"{json_file} does not contain class_id / predicted_class / pred_class / cls / morphology_label field."
            )

        return _getter

    else:
        raise ValueError(f"--class_json path does not exist: {class_json_path}")


def extract_image_identifier(image_obj):
    if isinstance(image_obj, str):
        return image_obj

    for attr in ("filename", "path", "name"):
        value = getattr(image_obj, attr, None)
        if isinstance(value, str) and len(value) > 0:
            return value

    if isinstance(image_obj, dict):
        for key in ("path", "file_name", "filename", "name"):
            value = image_obj.get(key)
            if isinstance(value, str) and len(value) > 0:
                return value

    raise ValueError(f"Cannot extract filename from type {type(image_obj)}, please check dataset column or add path field.")


def save_moe_experts(model, output_dir):
    moe_dir = os.path.join(output_dir, "moe_unet")
    os.makedirs(moe_dir, exist_ok=True)
    torch.save(
        {
            "num_experts": model.num_classes,
            "experts_state_dict": model.experts.state_dict(),
        },
        os.path.join(moe_dir, "experts.pt"),
    )


def load_moe_experts(model, input_dir):
    expert_ckpt = os.path.join(input_dir, "moe_unet", "experts.pt")
    if not os.path.exists(expert_ckpt):
        logger.warning(f"No MoE expert checkpoint found at {expert_ckpt}, skip loading experts.")
        return

    state = torch.load(expert_ckpt, map_location="cpu")
    state_dict = state.get("experts_state_dict", state)
    model.experts.load_state_dict(state_dict)


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def download_image(url):
    if os.path.exists(url): 
        image = PIL.Image.open(url)
    else: 
        image = PIL.Image.open(requests.get(url, stream=True, timeout=DIFFUSERS_REQUEST_TIMEOUT).raw)

    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def main():
    args = parse_args()
    for k in [
        "pretrained_model_name_or_path",
        "unet_model_name_or_path",
        "dataset_name",
        "prior_dir",
        "sdf_dir",
        "class_json",
        "num_experts",
        "output_dir",
    ]:
        print(f"{k} = {getattr(args, k, None)}", flush=True)
    print("===== DEBUG ARGS END =====", flush=True)
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.seed is not None:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    else:
        generator = torch.Generator(device=accelerator.device)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet_path = args.unet_model_name_or_path if args.unet_model_name_or_path else args.pretrained_model_name_or_path
    unet = UNet2DConditionModel.from_pretrained(
        unet_path, subfolder="unet", revision=args.non_ema_revision
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()
    if hasattr(args, "use_ema") and args.use_ema:
        args.use_ema = False
        logger.warning("EMA is not supported for ControlNet training in this simple script yet. Disabled EMA.")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    unet = MoEUNetWrapper(unet, num_classes=args.num_experts)
    for p in unet.experts.parameters():
        p.requires_grad = True
    unet.train()
   
    print("type(unet) =", type(unet), flush=True)
    print("hasattr(unet, 'experts') =", hasattr(unet, "experts"), flush=True)
    if hasattr(unet, "experts"):
        print("num experts =", len(unet.experts), flush=True)
        for i, expert in enumerate(unet.experts):
            print(f"expert[{i}] = {expert}", flush=True)


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                while len(weights) > 0:
                    weights.pop()
                    model = models.pop()
                    unwrapped = unwrap_model(model)

                    if isinstance(unwrapped, ControlNetModel):
                        unwrapped.save_pretrained(os.path.join(output_dir, "controlnet"))
                    elif isinstance(unwrapped, MoEUNetWrapper):
                        save_moe_experts(unwrapped, output_dir)

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                unwrapped = unwrap_model(model)

                if isinstance(unwrapped, ControlNetModel):
                    load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                    unwrapped.register_to_config(**load_model.config)
                    unwrapped.load_state_dict(load_model.state_dict())
                    del load_model
                elif isinstance(unwrapped, MoEUNetWrapper):
                    load_moe_experts(unwrapped, input_dir)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params_to_optimize = list(controlnet.parameters()) + list(unet.experts.parameters())
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    print("controlnet trainable params =", sum(p.numel() for p in controlnet.parameters() if p.requires_grad), flush=True)
    print("unet expert params =", sum(p.numel() for p in unet.experts.parameters() if p.requires_grad), flush=True)


    if args.dataset_name is not None:
        metadata_path = os.path.join(args.dataset_name, "metadata.jsonl") if os.path.isdir(args.dataset_name) else None
        if metadata_path and os.path.exists(metadata_path):
            logger.info(f"Loading local dataset as json from {metadata_path}")
            dataset = load_dataset("json", data_files={"train": metadata_path}, cache_dir=args.cache_dir)
        else:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )

        from datasets import Image
        dataset["train"] = dataset["train"].cast_column(args.original_image_column, Image())
        dataset["train"] = dataset["train"].cast_column(args.edited_image_column, Image())
        def _find_none(ds, col, path_col_fallbacks=("file_name","edited_image","input_image")):
            bad = []
            for i in range(min(len(ds), 2000)):
                x = ds[i][col]
                if x is None:
                    info = {"idx": i, "col": col}
                    for k in path_col_fallbacks:
                        if k in ds.column_names:
                            info[k] = ds[i].get(k)
                    bad.append(info)
                    if len(bad) >= 20:
                        break
            return bad

        bad_o = _find_none(dataset["train"], args.original_image_column)
        bad_e = _find_none(dataset["train"], args.edited_image_column)

        print("None in original:", len(bad_o), bad_o[:5], flush=True)
        print("None in edited  :", len(bad_e), bad_e[:5], flush=True)
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )

    column_names = dataset["train"].column_names

    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.original_image_column is None:
        original_image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        original_image_column = args.original_image_column
        if original_image_column not in column_names:
            raise ValueError(
                f"--original_image_column' value '{args.original_image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edit_prompt_column is None:
        edit_prompt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        edit_prompt_column = args.edit_prompt_column
        if edit_prompt_column not in column_names:
            raise ValueError(
                f"--edit_prompt_column' value '{args.edit_prompt_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edited_image_column is None:
        edited_image_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        edited_image_column = args.edited_image_column
        if edited_image_column not in column_names:
            raise ValueError(
                f"--edited_image_column' value '{args.edited_image_column}' needs to be one of: {', '.join(column_names)}"
            )

    class_id_getter = build_class_id_getter(args.class_json)

    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    def preprocess_images(examples):
        original_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[original_image_column]]
        )
        edited_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[edited_image_column]]
        )
        images = np.stack([original_images, edited_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return train_transforms(images)

    def preprocess_train(examples):
        preprocessed_images = preprocess_images(examples)
        original_images, edited_images = preprocessed_images
        original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)
        examples["original_pixel_values"] = original_images
        examples["edited_pixel_values"] = edited_images

        captions = list(examples[edit_prompt_column])
        examples["input_ids"] = tokenize_captions(captions)

        class_ids = []
        for img in examples[original_image_column]:
            image_id = extract_image_identifier(img)
            class_ids.append(class_id_getter(image_id))
        examples["class_id"] = class_ids

        if args.prior_dir is not None:
            prior_tensors = []
            for img in examples[original_image_column]:
                image_id = extract_image_identifier(img)
                base_name = os.path.basename(image_id)

                prefix = base_name.split("_")[0]
                prior_path = os.path.join(args.prior_dir, f"{prefix}_seg.npy")
                
                if not os.path.exists(prior_path):
                    raise FileNotFoundError(f"Prior file missing for {base_name}: {prior_path}")

                prior_np = np.load(prior_path)
                prior_tensor = torch.from_numpy(prior_np).float()
                
                latent_size = args.resolution // 8
                prior_tensor = prior_tensor.unsqueeze(0)
                prior_tensor = torch.nn.functional.interpolate(
                    prior_tensor, size=(latent_size, latent_size), mode="bilinear", align_corners=False
                )
                prior_tensor = prior_tensor.squeeze(0)
                prior_tensors.append(prior_tensor)

            examples["prior_pixel_values"] = prior_tensors
            
        if args.sdf_dir is not None:
            sdf_tensors = []
            for img in examples[original_image_column]:
                image_id = extract_image_identifier(img)
                base_name = os.path.basename(image_id)

                prefix = base_name.split("_")[0]
                sdf_path = os.path.join(args.sdf_dir, f"{prefix}_sdf.npy")
                if not os.path.exists(sdf_path):
                    raise FileNotFoundError(f"SDF file missing for {base_name}: {sdf_path}")

                sdf_np = np.load(sdf_path)
                sdf_tensor = torch.from_numpy(sdf_np).float()
                if sdf_tensor.shape[0] == 1:
                    sdf_tensor = sdf_tensor.repeat(3, 1, 1)

                sdf_tensor = sdf_tensor.unsqueeze(0)
                sdf_tensor = torch.nn.functional.interpolate(
                    sdf_tensor, size=(args.resolution, args.resolution), mode="bilinear", align_corners=False
                )
                sdf_tensor = sdf_tensor.squeeze(0)
                sdf_tensors.append(sdf_tensor)
                
            examples["sdf_pixel_values"] = sdf_tensors

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
        edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        
        class_id = torch.tensor([int(example["class_id"]) for example in examples], dtype=torch.long)

        batch = {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "input_ids": input_ids,
            "class_id": class_id,
        }

        if args.prior_dir is not None:
            prior_pixel_values = torch.stack([example["prior_pixel_values"] for example in examples])
            batch["prior_pixel_values"] = prior_pixel_values

        if args.sdf_dir is not None:
            sdf_pixel_values = torch.stack([example["sdf_pixel_values"] for example in examples])
            batch["sdf_pixel_values"] = sdf_pixel_values

        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    unet, controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, controlnet, optimizer, train_dataloader, lr_scheduler
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("sgdiff", config=vars(args))
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(controlnet):
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    original_image_embeds = image_mask * original_image_embeds

                if args.prior_dir is not None:
                    prior_mask = batch["prior_pixel_values"].to(weight_dtype)
                    concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds, prior_mask], dim=1)
                else:
                    concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                if args.sdf_dir is not None:
                    controlnet_image = batch["sdf_pixel_values"].to(weight_dtype)
                else:
                    controlnet_image = torch.zeros((bsz, 3, args.resolution, args.resolution), device=latents.device, dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    concatenated_noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = unet(
                    concatenated_noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    class_id=batch["class_id"],
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(controlnet.parameters()) + [p for p in unet.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        if accelerator.is_main_process:
                            unwrapped_unet = unwrap_model(unet)
                            moe_dir = os.path.join(save_path, "moe_unet")
                            os.makedirs(moe_dir, exist_ok=True)
                            torch.save(
                                {
                                    "num_experts": unwrapped_unet.num_classes,
                                    "experts_state_dict": unwrapped_unet.experts.state_dict(),
                                },
                                os.path.join(moe_dir, "experts.pt"),
                            )
                            print(f"DEBUG_EXPERT_SAVE: {os.path.join(moe_dir, 'experts.pt')}", flush=True)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if (
                (args.val_image_url is not None)
                and (args.validation_prompt is not None)
                and (epoch % args.validation_epochs == 0)
            ):
                logger.warning("Validation flow in ControlNet training not fully implemented for IP2P.")
                pass

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        unet = unwrap_model(unet)
        print("DEBUG_SAVE_MOE_BEGIN", flush=True)
        print("DEBUG_SAVE: saving controlnet to", args.output_dir, flush=True)
        controlnet.save_pretrained(args.output_dir)
        if accelerator.is_main_process:
            unwrapped_unet = unwrap_model(unet)
            moe_dir = os.path.join(args.output_dir, "moe_unet")
            os.makedirs(moe_dir, exist_ok=True)
            torch.save(
                {
                    "num_experts": unwrapped_unet.num_classes,
                    "experts_state_dict": unwrapped_unet.experts.state_dict(),
                },
                os.path.join(moe_dir, "experts.pt"),
            )
            print(f"DEBUG_FINAL_EXPERT_SAVE: {os.path.join(moe_dir, 'experts.pt')}", flush=True)
        print("DEBUG_SAVE: saving moe experts to", args.output_dir, flush=True)
        save_moe_experts(unet, args.output_dir)
        print("DEBUG_SAVE: moe experts saved", flush=True)
        print("DEBUG_SAVE_MOE_END", flush=True)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
