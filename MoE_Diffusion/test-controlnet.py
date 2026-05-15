import argparse
import copy
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage import metrics
from tqdm import tqdm

from diffusers import (
    ControlNetModel,
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
)
from diffusers.configuration_utils import FrozenDict
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--unet_path", type=str, required=True)
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--experts_path", type=str, default=None)

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--prior_dir", type=str, required=True)
    parser.add_argument("--sdf_dir", type=str, required=True)
    parser.add_argument("--class_json_path", type=str, required=True)

    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--guidance_scale", type=float, default=7.0)

    return parser.parse_args()


class MoEUNetWrapper(nn.Module):
    def __init__(self, original_unet, num_classes=5, num_expert_up_blocks=2):
        super().__init__()
        self.unet = original_unet
        self.num_classes = num_classes
        self.num_expert_up_blocks = num_expert_up_blocks

        if len(self.unet.up_blocks) < num_expert_up_blocks:
            raise ValueError(
                f"UNet has only {len(self.unet.up_blocks)} up blocks, cannot make "
                f"{num_expert_up_blocks} expert-specific."
            )

        self.shared_up_block_count = len(self.unet.up_blocks) - num_expert_up_blocks
        expert_template_blocks = self.unet.up_blocks[self.shared_up_block_count:]
        self.experts = nn.ModuleList([
            nn.ModuleList([copy.deepcopy(block) for block in expert_template_blocks])
            for _ in range(num_classes)
        ])

        for block in expert_template_blocks:
            block.requires_grad_(False)

    def _expand_timesteps(self, timesteps, batch_size, device):
        if not torch.is_tensor(timesteps):
            is_mps = device.type == "mps"
            dtype = torch.float32 if isinstance(timesteps, float) else torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
        return timesteps.expand(batch_size)

    def _prepare_attention_mask(self, attention_mask, dtype):
        if attention_mask is None:
            return None
        attention_mask = (1 - attention_mask.to(dtype)) * -10000.0
        return attention_mask.unsqueeze(1)

    def _run_down_blocks(
        self,
        sample,
        emb,
        encoder_hidden_states,
        attention_mask,
        cross_attention_kwargs,
        encoder_attention_mask,
        down_block_additional_residuals,
        down_intrablock_additional_residuals,
    ):
        down_block_res_samples = (sample,)
        intrablock_residuals = list(down_intrablock_additional_residuals) if down_intrablock_additional_residuals is not None else None

        for downsample_block in self.unet.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals = None
                if intrablock_residuals is not None and len(intrablock_residuals) > 0:
                    additional_residuals = intrablock_residuals.pop(0)

                block_kwargs = {
                    "hidden_states": sample,
                    "temb": emb,
                    "encoder_hidden_states": encoder_hidden_states,
                    "attention_mask": attention_mask,
                    "cross_attention_kwargs": cross_attention_kwargs,
                    "encoder_attention_mask": encoder_attention_mask,
                }
                if additional_residuals is not None:
                    block_kwargs["additional_residuals"] = additional_residuals
                try:
                    sample, res_samples = downsample_block(**block_kwargs)
                except TypeError:
                    block_kwargs.pop("encoder_attention_mask", None)
                    sample, res_samples = downsample_block(**block_kwargs)
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if intrablock_residuals is not None and len(intrablock_residuals) > 0:
                    sample = sample + intrablock_residuals.pop(0)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            down_block_res_samples = tuple(
                down_block_res_sample + down_block_additional_residual
                for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
                )
            )

        return sample, down_block_res_samples

    def _run_mid_block(
        self,
        sample,
        emb,
        encoder_hidden_states,
        attention_mask,
        cross_attention_kwargs,
        encoder_attention_mask,
        mid_block_additional_residual,
    ):
        if self.unet.mid_block is not None:
            if hasattr(self.unet.mid_block, "has_cross_attention") and self.unet.mid_block.has_cross_attention:
                mid_kwargs = {
                    "encoder_hidden_states": encoder_hidden_states,
                    "attention_mask": attention_mask,
                    "cross_attention_kwargs": cross_attention_kwargs,
                    "encoder_attention_mask": encoder_attention_mask,
                }
                try:
                    sample = self.unet.mid_block(sample, emb, **mid_kwargs)
                except TypeError:
                    mid_kwargs.pop("encoder_attention_mask", None)
                    sample = self.unet.mid_block(sample, emb, **mid_kwargs)
            else:
                sample = self.unet.mid_block(sample, emb)

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        return sample

    def _run_up_block(
        self,
        upsample_block,
        sample,
        res_samples,
        emb,
        encoder_hidden_states,
        attention_mask,
        cross_attention_kwargs,
        encoder_attention_mask,
        upsample_size,
    ):
        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            block_kwargs = {
                "hidden_states": sample,
                "temb": emb,
                "res_hidden_states_tuple": res_samples,
                "encoder_hidden_states": encoder_hidden_states,
                "cross_attention_kwargs": cross_attention_kwargs,
                "upsample_size": upsample_size,
                "attention_mask": attention_mask,
                "encoder_attention_mask": encoder_attention_mask,
            }
            try:
                return upsample_block(**block_kwargs)
            except TypeError:
                block_kwargs.pop("encoder_attention_mask", None)
                return upsample_block(**block_kwargs)

        return upsample_block(
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=res_samples,
            upsample_size=upsample_size,
        )

    def _select_batch(self, value, mask):
        if value is None or not torch.is_tensor(value) or value.shape[0] != mask.shape[0]:
            return value
        return value[mask]

    def _run_expert_up_blocks(
        self,
        sample,
        down_block_res_samples,
        emb,
        encoder_hidden_states,
        attention_mask,
        cross_attention_kwargs,
        encoder_attention_mask,
        class_id,
        forward_upsample_size,
    ):
        expert_sample = torch.empty_like(sample)

        for cid in torch.unique(class_id).tolist():
            cid = int(cid)
            if cid < 0 or cid >= self.num_classes:
                raise ValueError(f"class_id={cid} out of range, should be in [0, {self.num_classes - 1}].")

            mask = class_id == cid
            hidden_states = sample[mask]
            class_down_samples = tuple(res[mask] for res in down_block_res_samples)
            class_emb = emb[mask]
            class_encoder_hidden_states = self._select_batch(encoder_hidden_states, mask)
            class_attention_mask = self._select_batch(attention_mask, mask)
            class_encoder_attention_mask = self._select_batch(encoder_attention_mask, mask)

            for local_i, upsample_block in enumerate(self.experts[cid]):
                global_i = self.shared_up_block_count + local_i
                is_final_block = global_i == len(self.unet.up_blocks) - 1
                res_samples = class_down_samples[-len(upsample_block.resnets):]
                class_down_samples = class_down_samples[:-len(upsample_block.resnets)]

                upsample_size = None
                if not is_final_block and forward_upsample_size and len(class_down_samples) > 0:
                    upsample_size = class_down_samples[-1].shape[2:]

                hidden_states = self._run_up_block(
                    upsample_block,
                    hidden_states,
                    res_samples,
                    class_emb,
                    class_encoder_hidden_states,
                    class_attention_mask,
                    cross_attention_kwargs,
                    class_encoder_attention_mask,
                    upsample_size,
                )

            expert_sample[mask] = hidden_states.to(dtype=expert_sample.dtype)

        return expert_sample

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        class_id=None,
        cross_attention_kwargs=None,
        attention_mask=None,
        encoder_attention_mask=None,
        timestep_cond=None,
        class_labels=None,
        added_cond_kwargs=None,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        down_intrablock_additional_residuals=None,
        return_dict=True,
        **kwargs
    ):
        cross_attention_kwargs = dict(cross_attention_kwargs) if cross_attention_kwargs is not None else None
        if class_id is None and cross_attention_kwargs is not None:
            if "class_id" in cross_attention_kwargs:
                 class_id = cross_attention_kwargs.pop("class_id")

        if class_id is None:
            raise ValueError("MoE inference must pass in the class_id.")

        if not torch.is_tensor(class_id):
            class_id = torch.tensor(class_id, device=sample.device)

        class_id = class_id.to(device=sample.device, dtype=torch.long).view(-1)

        if class_id.shape[0] != sample.shape[0]:
            if sample.shape[0] % class_id.shape[0] == 0:
                repeat_factor = sample.shape[0] // class_id.shape[0]
                class_id = class_id.repeat_interleave(repeat_factor)
            else:
                raise ValueError(
                    f"class_id batch size ({class_id.shape[0]}) does not match sample batch size ({sample.shape[0]})."
                )

        default_overall_up_factor = 2 ** getattr(self.unet, "num_upsamplers", 0)
        forward_upsample_size = False
        if default_overall_up_factor > 0:
            for dim in sample.shape[-2:]:
                if dim % default_overall_up_factor != 0:
                    forward_upsample_size = True
                    break

        attention_mask = self._prepare_attention_mask(attention_mask, sample.dtype)
        encoder_attention_mask = self._prepare_attention_mask(encoder_attention_mask, sample.dtype)

        if getattr(self.unet.config, "center_input_sample", False):
            sample = 2 * sample - 1.0

        batch_size = sample.shape[0]
        timesteps = self._expand_timesteps(timestep, batch_size, sample.device)
        t_emb = self.unet.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        try:
            emb = self.unet.time_embedding(t_emb, timestep_cond)
        except TypeError:
            emb = self.unet.time_embedding(t_emb)

        if getattr(self.unet, "class_embedding", None) is not None:
            if class_labels is None:
                class_labels = class_id
            if getattr(self.unet.config, "class_embed_type", None) == "timestep":
                class_labels = self.unet.time_proj(class_labels)
                class_labels = class_labels.to(dtype=sample.dtype)
            class_emb = self.unet.class_embedding(class_labels).to(dtype=sample.dtype)
            emb = emb + class_emb

        aug_emb = None
        if getattr(self.unet, "add_embedding", None) is not None and added_cond_kwargs is not None:
            if getattr(self.unet.config, "addition_embed_type", None) == "text":
                aug_emb = self.unet.add_embedding(encoder_hidden_states)
        if aug_emb is not None:
            emb = emb + aug_emb

        if getattr(self.unet, "time_embed_act", None) is not None:
            emb = self.unet.time_embed_act(emb)

        if getattr(self.unet, "encoder_hid_proj", None) is not None:
            encoder_hidden_states = self.unet.encoder_hid_proj(encoder_hidden_states)

        sample = self.unet.conv_in(sample)
        sample, down_block_res_samples = self._run_down_blocks(
            sample,
            emb,
            encoder_hidden_states,
            attention_mask,
            cross_attention_kwargs,
            encoder_attention_mask,
            down_block_additional_residuals,
            down_intrablock_additional_residuals,
        )

        sample = self._run_mid_block(
            sample,
            emb,
            encoder_hidden_states,
            attention_mask,
            cross_attention_kwargs,
            encoder_attention_mask,
            mid_block_additional_residual,
        )

        for i, upsample_block in enumerate(self.unet.up_blocks[:self.shared_up_block_count]):
            is_final_block = i == len(self.unet.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            upsample_size = None
            if not is_final_block and forward_upsample_size and len(down_block_res_samples) > 0:
                upsample_size = down_block_res_samples[-1].shape[2:]

            sample = self._run_up_block(
                upsample_block,
                sample,
                res_samples,
                emb,
                encoder_hidden_states,
                attention_mask,
                cross_attention_kwargs,
                encoder_attention_mask,
                upsample_size,
            )

        sample = self._run_expert_up_blocks(
            sample,
            down_block_res_samples,
            emb,
            encoder_hidden_states,
            attention_mask,
            cross_attention_kwargs,
            encoder_attention_mask,
            class_id,
            forward_upsample_size,
        )

        if getattr(self.unet, "conv_norm_out", None) is not None:
            sample = self.unet.conv_norm_out(sample)
            sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)

        if return_dict:
            return UNet2DConditionOutput(sample=sample)
        return (sample,)
    @property
    def dtype(self):
        return self.unet.dtype

    @property
    def config(self):
        return self.unet.config


def load_moe_experts(model, experts_path):
    if not os.path.exists(experts_path):
        raise FileNotFoundError(f"experts.pt not found: {experts_path}")

    ckpt = torch.load(experts_path, map_location="cpu")
    state_dict = ckpt.get("experts_state_dict", ckpt)
    try:
        model.experts.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "MoE expert checkpoint is incompatible with the current last-two-decoder-block expert architecture. "
        ) from exc
    print(f"MoE experts loaded from: {experts_path}")


def build_class_id_getter(class_json_path):
    """
    Supports two forms:
    1) class_json_path is a total mapping json
    2) class_json_path is a directory, with each image having a json
    """
    if class_json_path is None:
        raise ValueError("Test phase requires class_json_path, cannot be empty.")

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

            raise KeyError(f"Cannot find class_id for image {image_name} in class_json mapping.")

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
                    f"Cannot find class json for {image_name}. Tried: {[str(x) for x in candidates]}"
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
        raise ValueError(f"class_json_path path does not exist: {class_json_path}")


def get_full_tooth_mask(img_gt_np):
    gray = (img_gt_np * 255).astype(np.uint8)
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    full_mask = np.zeros_like(binary)
    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(full_mask, [max_cnt], -1, 255, thickness=cv2.FILLED)
    return full_mask


def calculate_ssim(img_gt, img_restored, mask_uint8):
    mask_f = mask_uint8.astype(np.float32) / 255.0
    _, ssim_map = metrics.structural_similarity(
        img_gt, img_restored, data_range=1, channel_axis=2, full=True
    )
    ssim_map_avg = np.mean(ssim_map, axis=2)
    return np.sum(ssim_map_avg * mask_f) / (np.sum(mask_f) + 1e-8)


def PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 100 if mse == 0 else 10 * math.log10(1 / mse)


def npy_to_bchw(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        pass
    elif arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    else:
        raise ValueError(f"Unsupported npy dimension: {arr.shape}")
    return torch.from_numpy(arr).float().unsqueeze(0)


args = parse_args()

model_path = args.model_path
unet_path = args.unet_path
controlnet_path = args.controlnet_path
experts_path = args.experts_path or os.path.join(controlnet_path, "moe_unet", "experts.pt")

data_root = args.data_root
jsonl_path = os.path.join(data_root, "metadata.jsonl")

prior_dir = args.prior_dir
sdf_dir = args.sdf_dir
class_json_path = args.class_json_path

class_id_getter = build_class_id_getter(class_json_path)

SAVE_DIR = args.save_dir
os.makedirs(SAVE_DIR, exist_ok=True)
REPORT_FILE = os.path.join(SAVE_DIR, "metrics_report.txt")

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

base_unet = UNet2DConditionModel.from_pretrained(
    unet_path,
    subfolder="unet",
    torch_dtype=torch.float16
).to(device)

moe_unet = MoEUNetWrapper(base_unet, num_classes=5).to(device)
load_moe_experts(moe_unet, experts_path)
pipe.unet = moe_unet

controlnet_load_path = os.path.join(controlnet_path, "controlnet")
if not os.path.exists(controlnet_load_path):
    controlnet_load_path = controlnet_path

controlnet = ControlNetModel.from_pretrained(
    controlnet_load_path,
    torch_dtype=torch.float16
).to(device)

current_prior_tensor = None
current_sdf_tensor = None


class PatchedUNetWithControlNet:
    def __init__(self, unet, controlnet):
        self.unet = unet
        self.controlnet = controlnet

        config_dict = dict(unet.config)
        config_dict["in_channels"] = 8
        self.config = FrozenDict(config_dict)

    def __getattr__(self, name):
        if name in ["unet", "controlnet"]:
            return super().__getattribute__(name)
        return getattr(self.unet, name)

    def __call__(self, sample, timestep, encoder_hidden_states, **kwargs):
        global current_prior_tensor, current_sdf_tensor

        if current_prior_tensor is not None:
            B = sample.shape[0]
            prior = current_prior_tensor.to(device=sample.device, dtype=sample.dtype)
            prior_batch = prior.repeat(B, 1, 1, 1)
            sample = torch.cat([sample, prior_batch], dim=1)

        if current_sdf_tensor is not None:
            B = sample.shape[0]
            sdf_cond = current_sdf_tensor.to(device=sample.device, dtype=sample.dtype)
            sdf_cond = sdf_cond.repeat(B, 3, 1, 1)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=sdf_cond,
                return_dict=False,
            )

            kwargs["down_block_additional_residuals"] = [
                res.to(self.unet.dtype) for res in down_block_res_samples
            ]
            kwargs["mid_block_additional_residual"] = mid_block_res_sample.to(self.unet.dtype)

        return self.unet(sample, timestep, encoder_hidden_states, **kwargs)

    @property
    def dtype(self):
        return self.unet.dtype


pipe.unet = PatchedUNetWithControlNet(pipe.unet, controlnet)

all_psnr, all_ssim = [], []
prompt = "remove metal artifacts"

with open(jsonl_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Starting evaluation of {len(lines)} groups of images...")

with open(REPORT_FILE, "w", encoding="utf-8") as report:
    report.write("SG-Diff Evaluation Report\n")
    report.write("-" * 50 + "\n")

    for i, line in enumerate(tqdm(lines)):
        data = json.loads(line)

        input_pil = Image.open(os.path.join(data_root, data["file_name"])).convert("RGB").resize((256, 256))
        target_pil = Image.open(os.path.join(data_root, data["edited_image"])).convert("RGB").resize((256, 256))

        base_name = os.path.basename(data["file_name"])
        prefix = base_name.split("_")[0]

        class_id = class_id_getter(data["file_name"])

        prior_path = os.path.join(prior_dir, f"{prefix}_seg.npy")
        if not os.path.exists(prior_path):
            raise FileNotFoundError(f"Prior file not found: {prior_path}")

        prior_np = np.load(prior_path)
        prior_tensor = npy_to_bchw(prior_np)

        latent_size = 256 // 8
        prior_tensor = torch.nn.functional.interpolate(
            prior_tensor, size=(latent_size, latent_size), mode="bilinear", align_corners=False
        )
        current_prior_tensor = prior_tensor

        sdf_path = os.path.join(sdf_dir, f"{prefix}_sdf.npy")
        if not os.path.exists(sdf_path):
            raise FileNotFoundError(f"SDF file not found: {sdf_path}")

        sdf_np = np.load(sdf_path)
        sdf_tensor = npy_to_bchw(sdf_np)
        sdf_tensor = torch.nn.functional.interpolate(
            sdf_tensor, size=(256, 256), mode="bilinear", align_corners=False
        )
        current_sdf_tensor = sdf_tensor

        if device == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda")
        else:
            autocast_ctx = torch.autocast(device_type="cpu")

        with autocast_ctx:
            output_pil = pipe(
                prompt,
                image=input_pil,
                num_inference_steps=args.num_inference_steps,
                image_guidance_scale=args.image_guidance_scale,
                guidance_scale=args.guidance_scale,
                cross_attention_kwargs={
                    "class_id": torch.tensor([class_id], device=device, dtype=torch.long)
                },
            ).images[0].resize((256, 256))

        input_np = np.array(input_pil)
        output_np = np.array(output_pil)
        target_np = np.array(target_pil)

        out_f = output_np.astype(np.float32) / 255.0
        gt_f = target_np.astype(np.float32) / 255.0

        mask_uint8 = get_full_tooth_mask(gt_f)
        cur_psnr = PSNR(gt_f, out_f)
        cur_ssim = calculate_ssim(gt_f, out_f, mask_uint8)

        all_psnr.append(cur_psnr)
        all_ssim.append(cur_ssim)

        report.write(
            f"Idx: {i:03d} | PSNR: {cur_psnr:.2f} | "
            f"SSIM: {cur_ssim:.4f} | {data['file_name']}\n"
        )

    summary = (
        f"\n{'='*60}\n"
        f"Final Statistics ({len(lines)} pairs):\n"
        f"Average PSNR:        {np.mean(all_psnr):.4f}\n"
        f"Average SSIM: {np.mean(all_ssim):.4f}\n"
        f"{'='*60}\n"
    )
    print(summary)
    report.write(summary)

print(f"Evaluation completed!\nData report: {REPORT_FILE}")
