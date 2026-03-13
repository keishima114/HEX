#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_HEX_CHANNEL_NAMES = [
    "DAPI",
    "CD8",
    "Pan-Cytokeratin",
    "CD3e",
    "CD163",
    "CD20",
    "CD4",
    "FAP",
    "CD138",
    "CD11c",
    "CD66b",
    "aSMA",
    "CD68",
    "Ki67",
    "CD31",
    "Collagen IV",
    "Granzyme B",
    "MMP9",
    "PD-1",
    "CD44",
    "PD-L1",
    "E-cadherin",
    "LAG3",
    "Mac2/Galectin-3",
    "FOXP3",
    "CD14",
    "EpCAM",
    "CD21",
    "CD45",
    "MPO",
    "TCF-1",
    "ICOS",
    "Bcl-2",
    "HLA-E",
    "CD45RO",
    "VISTA",
    "HIF1A",
    "CD39",
    "CD40",
    "HLA-DR",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize spatial HEX marker predictions from per-slide H5 files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="HEX H5 path, directory, or glob pattern.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where PNGs and optional NPZ grids will be written.",
    )
    parser.add_argument(
        "--markers",
        nargs="+",
        required=True,
        help="Marker names or 1-based channel indices to visualize.",
    )
    parser.add_argument(
        "--channel-names",
        nargs="+",
        default=DEFAULT_HEX_CHANNEL_NAMES,
        help="Channel names in H5 column order. Defaults to the standard 40 HEX markers.",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap for heatmaps.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Upper percentile used for color scaling.",
    )
    parser.add_argument(
        "--min-percentile",
        type=float,
        default=0.0,
        help="Lower percentile used for color scaling.",
    )
    parser.add_argument(
        "--panel-cols",
        type=int,
        default=3,
        help="Number of columns for the combined marker panel.",
    )
    parser.add_argument(
        "--dot-size",
        type=float,
        default=10.0,
        help="Scatter dot size in points.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.95,
        help="Scatter alpha.",
    )
    parser.add_argument(
        "--invert-y",
        action="store_true",
        help="Invert Y axis so image space appears top-down.",
    )
    parser.add_argument(
        "--save-grid",
        action="store_true",
        help="Also save per-slide spatial grids as compressed NPZ files.",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Save PNGs with transparent background.",
    )
    return parser.parse_args()


def resolve_h5_paths(input_spec: str) -> List[Path]:
    path = Path(input_spec).expanduser()
    if path.exists():
        if path.is_dir():
            return sorted(path.glob("*.h5"))
        if path.suffix.lower() == ".h5":
            return [path]

    matches = sorted(Path(p).resolve() for p in glob.glob(input_spec, recursive=True))
    if matches:
        return [p for p in matches if p.suffix.lower() == ".h5"]
    raise FileNotFoundError(f"No H5 files found for input: {input_spec}")


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def resolve_markers(markers: Sequence[str], channel_names: Sequence[str]) -> List[Tuple[int, str]]:
    if len(channel_names) == 0:
        raise ValueError("channel_names is empty")

    name_to_idx = {normalize_name(name): i for i, name in enumerate(channel_names)}
    resolved: List[Tuple[int, str]] = []

    for marker in markers:
        key = marker.strip()
        if not key:
            continue
        try:
            idx1 = int(key)
            if idx1 < 1 or idx1 > len(channel_names):
                raise ValueError(f"Marker index out of range [1,{len(channel_names)}]: {idx1}")
            idx0 = idx1 - 1
            resolved.append((idx0, channel_names[idx0]))
            continue
        except ValueError:
            pass

        normalized = normalize_name(key)
        if normalized not in name_to_idx:
            raise ValueError(
                f"Unknown marker '{marker}'. Known markers: {', '.join(channel_names)}"
            )
        idx0 = name_to_idx[normalized]
        resolved.append((idx0, channel_names[idx0]))

    if not resolved:
        raise ValueError("No valid markers resolved from --markers")
    return resolved


def load_hex_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as handle:
        if "codex_prediction" not in handle or "coords" not in handle:
            raise ValueError(
                f"H5 must contain 'codex_prediction' and 'coords': {h5_path}"
            )
        predictions = np.asarray(handle["codex_prediction"][:], dtype=np.float32)
        coords = np.asarray(handle["coords"][:], dtype=np.int32)

    if predictions.ndim != 2:
        raise ValueError(f"'codex_prediction' must be 2D (N, C): {h5_path}")
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"'coords' must be shape (N, 2): {h5_path}")
    if predictions.shape[0] != coords.shape[0]:
        raise ValueError(
            f"Tile count mismatch between predictions and coords: {h5_path}"
        )
    return predictions, coords[:, :2]


def infer_step(values: np.ndarray) -> int:
    uniq = np.unique(values.astype(np.int64))
    if uniq.size <= 1:
        return 1
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1
    return int(np.gcd.reduce(diffs))


def build_spatial_grid(
    coords_xy: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = coords_xy[:, 0].astype(np.int64)
    ys = coords_xy[:, 1].astype(np.int64)
    min_x, min_y = int(xs.min()), int(ys.min())
    step_x = infer_step(xs)
    step_y = infer_step(ys)

    gx = ((xs - min_x) // step_x).astype(np.int64)
    gy = ((ys - min_y) // step_y).astype(np.int64)
    width = int(gx.max()) + 1
    height = int(gy.max()) + 1

    grid = np.full((height, width), np.nan, dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.int32)

    for x_idx, y_idx, value in zip(gx, gy, values, strict=False):
        if np.isnan(grid[y_idx, x_idx]):
            grid[y_idx, x_idx] = float(value)
        else:
            grid[y_idx, x_idx] += float(value)
        counts[y_idx, x_idx] += 1

    valid = counts > 0
    grid[valid] = grid[valid] / counts[valid]

    extent = np.array(
        [
            min_x - step_x / 2.0,
            min_x + step_x * width - step_x / 2.0,
            min_y + step_y * height - step_y / 2.0,
            min_y - step_y / 2.0,
        ],
        dtype=np.float32,
    )
    return grid, valid, extent


def compute_limits(values: np.ndarray, low_pct: float, high_pct: float) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, low_pct))
    vmax = float(np.percentile(finite, high_pct))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(finite))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def save_single_marker_plot(
    coords_xy: np.ndarray,
    values: np.ndarray,
    marker_name: str,
    out_path: Path,
    cmap: str,
    dot_size: float,
    alpha: float,
    low_pct: float,
    high_pct: float,
    invert_y: bool,
    transparent: bool,
) -> None:
    vmin, vmax = compute_limits(values, low_pct, high_pct)
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        coords_xy[:, 0],
        coords_xy[:, 1],
        c=values,
        s=dot_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        alpha=alpha,
    )
    ax.set_title(marker_name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    if invert_y:
        ax.invert_yaxis()
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("prediction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", transparent=transparent)
    plt.close(fig)


def save_marker_panel(
    slide_id: str,
    coords_xy: np.ndarray,
    marker_values: Sequence[Tuple[str, np.ndarray]],
    out_path: Path,
    cmap: str,
    dot_size: float,
    alpha: float,
    low_pct: float,
    high_pct: float,
    panel_cols: int,
    invert_y: bool,
    transparent: bool,
) -> None:
    n_markers = len(marker_values)
    n_cols = max(1, min(panel_cols, n_markers))
    n_rows = int(np.ceil(n_markers / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (marker_name, values) in zip(axes_flat, marker_values, strict=False):
        vmin, vmax = compute_limits(values, low_pct, high_pct)
        sc = ax.scatter(
            coords_xy[:, 0],
            coords_xy[:, 1],
            c=values,
            s=dot_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            alpha=alpha,
        )
        ax.set_title(marker_name)
        ax.set_aspect("equal")
        if invert_y:
            ax.invert_yaxis()
        fig.colorbar(sc, ax=ax, shrink=0.8)

    for ax in axes_flat[n_markers:]:
        ax.axis("off")

    fig.suptitle(slide_id)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", transparent=transparent)
    plt.close(fig)


def save_grid_npz(
    out_path: Path,
    marker_grids: Sequence[Tuple[str, np.ndarray]],
    valid_mask: np.ndarray,
    extent: np.ndarray,
) -> None:
    payload = {
        "marker_names": np.array([name for name, _ in marker_grids], dtype=object),
        "valid_mask": valid_mask.astype(bool),
        "extent": extent.astype(np.float32),
    }
    for name, grid in marker_grids:
        payload[f"grid::{name}"] = grid.astype(np.float32)
    np.savez_compressed(out_path, **payload)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_paths = resolve_h5_paths(args.input)
    markers = resolve_markers(args.markers, args.channel_names)

    for h5_path in h5_paths:
        predictions, coords_xy = load_hex_h5(h5_path)
        if predictions.shape[1] < len(args.channel_names):
            print(
                f"[warn] {h5_path.name}: H5 has {predictions.shape[1]} channels, "
                f"but {len(args.channel_names)} channel names were provided."
            )

        slide_id = h5_path.stem
        slide_dir = output_dir / slide_id
        slide_dir.mkdir(parents=True, exist_ok=True)

        panel_values: List[Tuple[str, np.ndarray]] = []
        grid_values: List[Tuple[str, np.ndarray]] = []
        shared_valid = None
        shared_extent = None

        for idx0, marker_name in markers:
            if idx0 >= predictions.shape[1]:
                raise ValueError(
                    f"Marker '{marker_name}' maps to channel {idx0 + 1}, "
                    f"but {h5_path.name} has only {predictions.shape[1]} channels."
                )
            values = predictions[:, idx0]
            panel_values.append((marker_name, values))
            save_single_marker_plot(
                coords_xy=coords_xy,
                values=values,
                marker_name=marker_name,
                out_path=slide_dir / f"{slide_id}_{marker_name}.png",
                cmap=args.cmap,
                dot_size=args.dot_size,
                alpha=args.alpha,
                low_pct=args.min_percentile,
                high_pct=args.percentile,
                invert_y=args.invert_y,
                transparent=args.transparent,
            )

            if args.save_grid:
                grid, valid, extent = build_spatial_grid(coords_xy, values)
                grid_values.append((marker_name, grid))
                if shared_valid is None:
                    shared_valid = valid
                    shared_extent = extent

        save_marker_panel(
            slide_id=slide_id,
            coords_xy=coords_xy,
            marker_values=panel_values,
            out_path=slide_dir / f"{slide_id}_panel.png",
            cmap=args.cmap,
            dot_size=args.dot_size,
            alpha=args.alpha,
            low_pct=args.min_percentile,
            high_pct=args.percentile,
            panel_cols=args.panel_cols,
            invert_y=args.invert_y,
            transparent=args.transparent,
        )

        if args.save_grid and grid_values and shared_valid is not None and shared_extent is not None:
            save_grid_npz(
                out_path=slide_dir / f"{slide_id}_spatial_grids.npz",
                marker_grids=grid_values,
                valid_mask=shared_valid,
                extent=shared_extent,
            )

        print(f"[done] {slide_id} -> {slide_dir}")


if __name__ == "__main__":
    main()
