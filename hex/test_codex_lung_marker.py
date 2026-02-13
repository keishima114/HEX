import argparse
import contextlib
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from hex.hex_architecture import CustomModel


TILE_NAME_PATTERN = re.compile(r"(?P<y>\d+)y_(?P<x>\d+)x$")


class TileDataset(Dataset):
    def __init__(self, tile_paths, transform=None):
        self.tile_paths = tile_paths
        self.transform = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        image_path = self.tile_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, str(image_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="HEX inference on tiled dataset with layout "
        "<data_dir>/<slide>/<slide>/<tile_name>.(jpeg|jpg|png)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/tishumap/data/SGH/Output/tiled_dataset_noresize",
        help="Root directory containing all slides.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=str((Path(__file__).resolve().parent / "checkpoint.pth")),
        help="Path to HEX checkpoint.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str((Path(__file__).resolve().parent / "inference_outputs")),
        help="Output directory for per-slide H5 predictions.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=["jpeg", "jpg", "png"],
        help="Image file extensions to scan (case-insensitive).",
    )
    parser.add_argument(
        "--local_only",
        action="store_true",
        help="Disable Hugging Face backbone loading and rely on local checkpoint only.",
    )
    return parser.parse_args()


def load_model(checkpoint_path, device, local_only=False):
    model = CustomModel(
        visual_output_dim=1024,
        num_outputs=40,
        load_musk_from_hf=not local_only,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    load_msg = model.load_state_dict(state_dict, strict=False)
    if len(load_msg.missing_keys) > 0:
        print(f"[warning] missing checkpoint keys: {len(load_msg.missing_keys)}")
        print(f"[warning] first 10 missing keys: {load_msg.missing_keys[:10]}")
    if len(load_msg.unexpected_keys) > 0:
        print(f"[warning] unexpected checkpoint keys: {len(load_msg.unexpected_keys)}")
        print(f"[warning] first 10 unexpected keys: {load_msg.unexpected_keys[:10]}")
    if device.type == "cuda":
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    model.eval()
    return model


def parse_tile_info(tile_path):
    slide_id = tile_path.parent.name
    match = TILE_NAME_PATTERN.search(tile_path.stem)
    if match is None:
        raise ValueError(
            f"Cannot parse tile coordinate from filename: {tile_path.name}. "
            "Expected '<y>y_<x>x.<ext>'."
        )
    y = int(match.group("y"))
    x = int(match.group("x"))
    return slide_id, x, y


def collect_tile_paths(data_dir, extensions):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    ext_set = {f".{e.lower().lstrip('.')}" for e in extensions}
    tile_paths = []
    for slide_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        nested_dir = slide_dir / slide_dir.name
        search_dir = nested_dir if nested_dir.exists() else slide_dir
        for path in search_dir.iterdir():
            if path.is_file() and path.suffix.lower() in ext_set:
                tile_paths.append(path)
    return tile_paths


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    h5_dir = save_dir / "codex_h5"
    h5_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading model from: {args.checkpoint_path}")
    if args.local_only:
        print("Local-only mode enabled: skipping Hugging Face MUSK weight loading.")
    model = load_model(args.checkpoint_path, device, local_only=args.local_only)

    print(f"Scanning tiles under: {args.data_dir}")
    tile_paths = collect_tile_paths(args.data_dir, args.extensions)
    if len(tile_paths) == 0:
        raise RuntimeError(f"No tiles found under {args.data_dir} with extensions {args.extensions}")
    print(f"Total tiles found: {len(tile_paths)}")

    transform = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
    )
    dataset = TileDataset(tile_paths, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    per_slide_preds = defaultdict(list)
    per_slide_coords = defaultdict(list)

    if device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = contextlib.nullcontext()

    print("Starting inference...")
    with torch.no_grad():
        with autocast_ctx:
            for images, image_paths in tqdm(dataloader, total=len(dataloader)):
                images = images.to(device, non_blocking=True)
                if device.type == "cuda":
                    dummy_labels = torch.zeros((images.shape[0], 40), device=device, dtype=torch.float16)
                else:
                    dummy_labels = torch.zeros((images.shape[0], 40), device=device, dtype=torch.float32)

                outputs, _ = model(images, dummy_labels, 0)
                preds = outputs.detach().float().cpu().numpy()

                for i, image_path in enumerate(image_paths):
                    slide_id, x, y = parse_tile_info(Path(image_path))
                    per_slide_preds[slide_id].append(preds[i])
                    per_slide_coords[slide_id].append((x, y))

    print("Saving per-slide H5 predictions...")
    summary_rows = []
    for slide_id in tqdm(sorted(per_slide_preds.keys())):
        preds = np.stack(per_slide_preds[slide_id]).astype(np.float32)
        coords = np.array(per_slide_coords[slide_id], dtype=np.int32)
        out_h5 = h5_dir / f"{slide_id}.h5"
        with h5py.File(out_h5, "w") as f:
            f.create_dataset("codex_prediction", data=preds, compression="gzip")
            f.create_dataset("coords", data=coords, compression="gzip")
        summary_rows.append({"slide_id": slide_id, "num_tiles": int(preds.shape[0]), "h5_path": str(out_h5)})

    summary_df = pd.DataFrame(summary_rows).sort_values("slide_id")
    summary_path = save_dir / "inference_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Done. Slides processed: {len(summary_rows)}")
    print(f"H5 outputs: {h5_dir}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
