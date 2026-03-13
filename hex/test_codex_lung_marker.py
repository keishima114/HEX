import argparse
import contextlib
import glob
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import load_file
from PIL import Image
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from hex.hex_architecture import CustomModel


TILE_NAME_PATTERN = re.compile(r"(?P<y>\d+)y_(?P<x>\d+)x$")


class TileDataset(Dataset):
    def __init__(self, tile_paths, transform=None, split_448_tiles=True):
        self.tile_paths = tile_paths
        self.transform = transform
        self.split_448_tiles = split_448_tiles

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        image_path = self.tile_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.split_448_tiles:
            if image.size != (448, 448):
                raise ValueError(
                    f"split_448_tiles=True expects 448x448 tiles, got {image.size} for {image_path}"
                )
            sub_tiles = [
                image.crop((0, 0, 224, 224)),
                image.crop((224, 0, 448, 224)),
                image.crop((0, 224, 224, 448)),
                image.crop((224, 224, 448, 448)),
            ]
            if self.transform is not None:
                sub_tiles = [self.transform(tile) for tile in sub_tiles]
            image = torch.stack(sub_tiles, dim=0)
        else:
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
    parser.add_argument(
        "--musk_weight_path",
        type=str,
        default=None,
        help="Local path to MUSK backbone weight file (e.g. model.safetensors).",
    )
    parser.add_argument(
        "--tile_list_path",
        type=str,
        default=None,
        help="Optional text file with one tile path per line. If it exists, scanning is skipped.",
    )
    parser.add_argument(
        "--no_split_448_tiles",
        action="store_false",
        dest="split_448_tiles",
        help="Disable 2x2 split inference for 448x448 tiles.",
    )
    parser.set_defaults(split_448_tiles=True)
    return parser.parse_args()


def load_model(checkpoint_path, device, local_only=False, musk_weight_path=None):
    model = CustomModel(
        visual_output_dim=1024,
        num_outputs=40,
        load_musk_from_hf=not local_only,
    )

    if musk_weight_path is not None:
        weight_path = Path(musk_weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"musk_weight_path does not exist: {weight_path}")
        print(f"Loading local MUSK backbone weights from: {weight_path}")
        try:
            # Available in some MUSK versions.
            from musk.modeling import fix_huggingface_weight_MUSK

            fixed_state = fix_huggingface_weight_MUSK(model.visual, str(weight_path))
            visual_msg = model.visual.load_state_dict(fixed_state, strict=False)
            if len(visual_msg.missing_keys) > 0:
                print(f"[warning] missing MUSK visual keys: {len(visual_msg.missing_keys)}")
                print(f"[warning] first 10 missing visual keys: {visual_msg.missing_keys[:10]}")
            if len(visual_msg.unexpected_keys) > 0:
                print(f"[warning] unexpected MUSK visual keys: {len(visual_msg.unexpected_keys)}")
                print(f"[warning] first 10 unexpected visual keys: {visual_msg.unexpected_keys[:10]}")
        except ImportError:
            # Fallback without MUSK utils/huggingface dependency: direct safetensors load.
            sd = load_file(str(weight_path))
            target_keys = set(model.visual.state_dict().keys())
            candidates = {
                "raw": sd,
                "strip_model": {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")},
                "strip_module": {k.replace("module.", "", 1): v for k, v in sd.items() if k.startswith("module.")},
            }

            best_name = "raw"
            best_match = -1
            for name, cand in candidates.items():
                match = sum(1 for k in cand.keys() if k in target_keys)
                if match > best_match:
                    best_name = name
                    best_match = match
            print(f"[info] local MUSK fallback mapping: {best_name} (matched keys: {best_match})")
            visual_msg = model.visual.load_state_dict(candidates[best_name], strict=False)
            if len(visual_msg.missing_keys) > 0:
                print(f"[warning] missing MUSK visual keys: {len(visual_msg.missing_keys)}")
                print(f"[warning] first 10 missing visual keys: {visual_msg.missing_keys[:10]}")
            if len(visual_msg.unexpected_keys) > 0:
                print(f"[warning] unexpected MUSK visual keys: {len(visual_msg.unexpected_keys)}")
                print(f"[warning] first 10 unexpected visual keys: {visual_msg.unexpected_keys[:10]}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
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


def collect_tile_paths(data_dir, extensions, tile_list_path=None):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    if tile_list_path is not None:
        tile_list_path = Path(tile_list_path)
        if tile_list_path.exists():
            print(f"Loading cached tile list from: {tile_list_path}")
            with tile_list_path.open("r") as f:
                paths = [line.strip() for line in f if line.strip()]
            return [Path(p) for p in paths]

    exts = [e.lower().lstrip(".") for e in extensions]
    tile_paths = []
    # Fast path for expected layout: <root>/<slide>/<slide>/<tile>.<ext>
    for ext in exts:
        tile_paths.extend(Path(p) for p in glob.iglob(str(data_dir / "*" / "*" / f"*.{ext}")))
        tile_paths.extend(Path(p) for p in glob.iglob(str(data_dir / "*" / "*" / f"*.{ext.upper()}")))
        # Fallback layout: <root>/<slide>/<tile>.<ext>
        tile_paths.extend(Path(p) for p in glob.iglob(str(data_dir / "*" / f"*.{ext}")))
        tile_paths.extend(Path(p) for p in glob.iglob(str(data_dir / "*" / f"*.{ext.upper()}")))

    # Deduplicate while preserving discovery order.
    seen = set()
    deduped = []
    for p in tile_paths:
        s = str(p)
        if s not in seen:
            seen.add(s)
            deduped.append(p)

    # Skip known non-tile directories (e.g., thumbnails) and filenames that
    # do not match the expected "<y>y_<x>x" naming.
    filtered = []
    skipped_thumbnail = 0
    skipped_name = 0
    for p in deduped:
        if any(part.lower() == "thumbnails" for part in p.parts):
            skipped_thumbnail += 1
            continue
        if TILE_NAME_PATTERN.search(p.stem) is None:
            skipped_name += 1
            continue
        filtered.append(p)

    if skipped_thumbnail > 0 or skipped_name > 0:
        print(
            f"Filtered tiles: kept={len(filtered)}, "
            f"skipped_thumbnails={skipped_thumbnail}, "
            f"skipped_bad_name={skipped_name}"
        )

    if tile_list_path is not None:
        tile_list_path.parent.mkdir(parents=True, exist_ok=True)
        with tile_list_path.open("w") as f:
            for p in filtered:
                f.write(f"{p}\n")
        print(f"Saved tile list cache to: {tile_list_path}")

    return filtered


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
    if args.musk_weight_path:
        print(f"Using local MUSK weight file: {args.musk_weight_path}")
    print(f"split_448_tiles: {args.split_448_tiles}")
    model = load_model(
        args.checkpoint_path,
        device,
        local_only=args.local_only,
        musk_weight_path=args.musk_weight_path,
    )

    print(f"Scanning tiles under: {args.data_dir}")
    tile_paths = collect_tile_paths(args.data_dir, args.extensions, args.tile_list_path)
    if len(tile_paths) == 0:
        raise RuntimeError(f"No tiles found under {args.data_dir} with extensions {args.extensions}")
    # Group by slide for incremental per-slide writes during inference.
    tile_paths = sorted(tile_paths, key=lambda p: p.parent.name)
    print(f"Total tiles found: {len(tile_paths)}")

    transform = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
    )
    dataset = TileDataset(
        tile_paths,
        transform=transform,
        split_448_tiles=args.split_448_tiles,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    per_slide_preds = defaultdict(list)
    per_slide_coords = defaultdict(list)
    summary_path = save_dir / "inference_summary.csv"
    if summary_path.exists():
        summary_path.unlink()
    summary_written_header = False

    def flush_slide(slide_id):
        nonlocal summary_written_header
        if slide_id is None or slide_id not in per_slide_preds:
            return
        if len(per_slide_preds[slide_id]) == 0:
            return
        preds = np.stack(per_slide_preds[slide_id]).astype(np.float32)
        coords = np.array(per_slide_coords[slide_id], dtype=np.int32)
        out_h5 = h5_dir / f"{slide_id}.h5"
        with h5py.File(out_h5, "w") as f:
            f.create_dataset("codex_prediction", data=preds, compression="gzip")
            f.create_dataset("coords", data=coords, compression="gzip")
        row = pd.DataFrame(
            [{"slide_id": slide_id, "num_tiles": int(preds.shape[0]), "h5_path": str(out_h5)}]
        )
        row.to_csv(summary_path, mode="a", index=False, header=not summary_written_header)
        summary_written_header = True
        # Free memory for completed slide.
        del per_slide_preds[slide_id]
        del per_slide_coords[slide_id]

    if device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = contextlib.nullcontext()

    print("Starting inference...")
    last_slide_id = None
    with torch.no_grad():
        with autocast_ctx:
            for images, image_paths in tqdm(dataloader, total=len(dataloader)):
                if args.split_448_tiles:
                    # images: [B, 4, C, H, W] -> [B*4, C, H, W], then average predictions back to [B, 40].
                    bsz, n_sub, channels, height, width = images.shape
                    flat_images = images.view(bsz * n_sub, channels, height, width).to(
                        device, non_blocking=True
                    )
                else:
                    bsz = images.shape[0]
                    n_sub = 1
                    flat_images = images.to(device, non_blocking=True)

                if device.type == "cuda":
                    dummy_labels = torch.zeros((flat_images.shape[0], 40), device=device, dtype=torch.float16)
                else:
                    dummy_labels = torch.zeros((flat_images.shape[0], 40), device=device, dtype=torch.float32)

                outputs, _ = model(flat_images, dummy_labels, 0)
                if n_sub > 1:
                    outputs = outputs.view(bsz, n_sub, -1).mean(dim=1)
                preds = outputs.detach().float().cpu().numpy()

                for i, image_path in enumerate(image_paths):
                    slide_id, x, y = parse_tile_info(Path(image_path))
                    if last_slide_id is None:
                        last_slide_id = slide_id
                    elif slide_id != last_slide_id:
                        flush_slide(last_slide_id)
                        last_slide_id = slide_id
                    per_slide_preds[slide_id].append(preds[i])
                    per_slide_coords[slide_id].append((x, y))

    # Flush any remaining slide(s).
    flush_slide(last_slide_id)
    for slide_id in list(per_slide_preds.keys()):
        flush_slide(slide_id)

    slides_processed = 0
    if summary_path.exists():
        slides_processed = len(pd.read_csv(summary_path))
    print(f"Done. Slides processed: {slides_processed}")
    print(f"H5 outputs: {h5_dir}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
