import os
import glob
import json

base_dir = "./datasets"
out_base = "./datasets-hf"

def main():
    artifactIma_files = glob.glob(os.path.join(base_dir, "artifactIma", "*.png"))
    cleanIma_files = glob.glob(os.path.join(base_dir, "cleanIma", "*.png"))

    artifactIma_dict = {os.path.basename(f)[:3]: f for f in artifactIma_files}
    cleanIma_dict = {os.path.basename(f)[:3]: f for f in cleanIma_files}

    for fold in range(5):
        for split in ["train", "test"]:
            txt_file = os.path.join(base_dir, "splits", f"fold{fold}_{split}.txt")
            if not os.path.exists(txt_file):
                continue

            with open(txt_file, 'r') as f:
                ids = [line.strip() for line in f if line.strip()]

            out_dir = os.path.join(out_base, f"fold{fold}", split)
            os.makedirs(out_dir, exist_ok=True)
            
            # create subfolders to keep it organized
            out_artifactIma = os.path.join(out_dir, "artifactIma")
            out_cleanIma = os.path.join(out_dir, "cleanIma")
            os.makedirs(out_artifactIma, exist_ok=True)
            os.makedirs(out_cleanIma, exist_ok=True)

            metadata = []
            for case_id in ids:
                if case_id in artifactIma_dict and case_id in cleanIma_dict:
                    src_with = artifactIma_dict[case_id]
                    src_no = cleanIma_dict[case_id]

                    tgt_with = os.path.join(out_artifactIma, os.path.basename(src_with))
                    tgt_no = os.path.join(out_cleanIma, os.path.basename(src_no))

                    if not os.path.exists(tgt_with):
                        os.symlink(src_with, tgt_with)
                    if not os.path.exists(tgt_no):
                        os.symlink(src_no, tgt_no)

                    metadata.append({
                        "file_name": tgt_with,
                        "edited_image": tgt_no,
                        "edit_prompt": "remove metal artifacts"
                    })

            # Save metadata
            meta_path = os.path.join(out_dir, "metadata.jsonl")
            with open(meta_path, "w") as f:
                for item in metadata:
                    f.write(json.dumps(item) + "\n")
            print(f"Prepared Fold {fold} {split}: {len(metadata)} images.")

    print(f"\n Dataset successfully exported to {out_base}")

if __name__ == "__main__":
    main()
