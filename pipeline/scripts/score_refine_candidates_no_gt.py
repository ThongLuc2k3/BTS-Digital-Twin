#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_gray01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def load_mask01(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=np.float32)
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if mask.shape != shape:
        raise SystemExit(f"Mask shape {mask.shape} khong khop image shape {shape}: {path}")
    return mask


def variance_of_laplacian(img: np.ndarray) -> float:
    center = -4.0 * img
    up = np.zeros_like(img)
    down = np.zeros_like(img)
    left = np.zeros_like(img)
    right = np.zeros_like(img)
    up[1:] = img[:-1]
    down[:-1] = img[1:]
    left[:, 1:] = img[:, :-1]
    right[:, :-1] = img[:, 1:]
    lap = center + up + down + left + right
    return float(lap.var())


def gradient_energy(img: np.ndarray) -> float:
    gy = np.diff(img, axis=0, prepend=img[:1])
    gx = np.diff(img, axis=1, prepend=img[:, :1])
    return float((gx * gx + gy * gy).mean())


def high_freq_ratio(img: np.ndarray) -> float:
    blur = (
        img
        + np.roll(img, 1, axis=0)
        + np.roll(img, -1, axis=0)
        + np.roll(img, 1, axis=1)
        + np.roll(img, -1, axis=1)
    ) / 5.0
    resid = np.abs(img - blur)
    return float(resid.mean() / max(img.mean(), 1e-6))


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def safe_crop(arr: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray | None:
    if box is None:
        return None
    x0, y0, x1, y1 = box
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return arr[y0:y1, x0:x1]


def percentile_norm(values: list[float], high_is_good: bool) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return []
    lo = float(np.percentile(arr, 5))
    hi = float(np.percentile(arr, 95))
    if hi <= lo:
        base = np.full_like(arr, 0.5, dtype=np.float64)
    else:
        base = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if high_is_good:
        return base.tolist()
    return (1.0 - base).tolist()


def auto_noise_mask(gray: np.ndarray, tower_mask: np.ndarray, skyline_top_frac: float) -> np.ndarray:
    h, w = gray.shape
    noise_mask = np.zeros((h, w), dtype=np.float32)
    top_h = max(int(round(h * skyline_top_frac)), 1)
    noise_mask[:top_h, :] = 1.0
    if tower_mask.max() > 0:
        noise_mask = np.clip(noise_mask - (tower_mask > 0.5).astype(np.float32), 0.0, 1.0)
    return noise_mask


def summarize_region(gray: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    box = mask_box(mask)
    crop = safe_crop(gray, box)
    if crop is None:
        return 0.0, 0.0, 0.0
    return variance_of_laplacian(crop), gradient_energy(crop), high_freq_ratio(crop)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--threshold_100", type=float, default=76.0)
    ap.add_argument("--tower_mask_dir", default=None)
    ap.add_argument("--noise_mask_dir", default=None)
    ap.add_argument("--skyline_top_frac", type=float, default=0.3)
    ap.add_argument("--summary_json", default=None)
    args = ap.parse_args()

    renders_dir = Path(args.renders_dir)
    out_csv = Path(args.out_csv)
    tower_mask_dir = Path(args.tower_mask_dir) if args.tower_mask_dir else None
    noise_mask_dir = Path(args.noise_mask_dir) if args.noise_mask_dir else None

    render_paths = sorted(p for p in renders_dir.glob("*.png") if p.is_file())
    if not render_paths:
        raise SystemExit(f"Khong thay render png nao trong {renders_dir}")

    raw_rows = []
    for render_path in render_paths:
        stem = render_path.stem
        gray = load_gray01(render_path)
        shape = gray.shape

        tower_mask = load_mask01(tower_mask_dir / f"{stem}.png", shape) if tower_mask_dir else np.zeros(shape, dtype=np.float32)
        if noise_mask_dir:
            noise_mask = load_mask01(noise_mask_dir / f"{stem}.png", shape)
        else:
            noise_mask = auto_noise_mask(gray, tower_mask, args.skyline_top_frac)

        global_lap = variance_of_laplacian(gray)
        global_grad = gradient_energy(gray)
        global_hf = high_freq_ratio(gray)

        tower_lap, tower_grad, tower_hf = summarize_region(gray, tower_mask)
        noise_lap, noise_grad, noise_hf = summarize_region(gray, noise_mask)

        tower_cov = float((tower_mask > 0.5).mean())
        noise_cov = float((noise_mask > 0.5).mean())

        raw_rows.append(
            {
                "image": stem,
                "global_lap": global_lap,
                "global_grad": global_grad,
                "global_hf": global_hf,
                "tower_lap": tower_lap,
                "tower_grad": tower_grad,
                "tower_hf": tower_hf,
                "tower_cov": tower_cov,
                "noise_lap": noise_lap,
                "noise_grad": noise_grad,
                "noise_hf": noise_hf,
                "noise_cov": noise_cov,
            }
        )

    norms = {
        "global_lap": percentile_norm([r["global_lap"] for r in raw_rows], high_is_good=True),
        "global_grad": percentile_norm([r["global_grad"] for r in raw_rows], high_is_good=True),
        "global_hf": percentile_norm([r["global_hf"] for r in raw_rows], high_is_good=False),
        "tower_lap": percentile_norm([r["tower_lap"] for r in raw_rows], high_is_good=True),
        "tower_grad": percentile_norm([r["tower_grad"] for r in raw_rows], high_is_good=True),
        "tower_hf": percentile_norm([r["tower_hf"] for r in raw_rows], high_is_good=False),
        "noise_lap": percentile_norm([r["noise_lap"] for r in raw_rows], high_is_good=True),
        "noise_grad": percentile_norm([r["noise_grad"] for r in raw_rows], high_is_good=True),
        "noise_hf": percentile_norm([r["noise_hf"] for r in raw_rows], high_is_good=False),
    }

    rows = []
    for idx, raw in enumerate(raw_rows):
        global_score = 100.0 * (
            0.55 * norms["global_lap"][idx]
            + 0.30 * norms["global_grad"][idx]
            + 0.15 * norms["global_hf"][idx]
        )

        if raw["tower_cov"] > 0.0:
            tower_score = 100.0 * (
                0.45 * norms["tower_lap"][idx]
                + 0.35 * norms["tower_grad"][idx]
                + 0.20 * norms["tower_hf"][idx]
            )
        else:
            tower_score = global_score

        if raw["noise_cov"] > 0.0:
            noise_score = 100.0 * (
                0.40 * norms["noise_lap"][idx]
                + 0.20 * norms["noise_grad"][idx]
                + 0.40 * norms["noise_hf"][idx]
            )
        else:
            noise_score = global_score

        proxy_score = (
            0.45 * global_score
            + 0.30 * tower_score
            + 0.25 * noise_score
        )

        refine_flag = int(proxy_score < args.threshold_100)
        priority = max(0.0, args.threshold_100 - proxy_score)
        rows.append(
            {
                **raw,
                "global_score_100": global_score,
                "tower_score_100": tower_score,
                "noise_score_100": noise_score,
                "proxy_score_100": proxy_score,
                "refine_flag": refine_flag,
                "refine_priority": priority,
            }
        )

    rows.sort(key=lambda r: (r["proxy_score_100"], r["image"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "proxy_score_100",
                "global_score_100",
                "tower_score_100",
                "noise_score_100",
                "refine_flag",
                "refine_priority",
                "tower_cov",
                "noise_cov",
                "global_lap",
                "global_grad",
                "global_hf",
                "tower_lap",
                "tower_grad",
                "tower_hf",
                "noise_lap",
                "noise_grad",
                "noise_hf",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    flagged = [r for r in rows if r["refine_flag"] == 1]
    summary = {
        "renders_dir": str(renders_dir),
        "out_csv": str(out_csv),
        "threshold_100": args.threshold_100,
        "images": len(rows),
        "flagged_images": len(flagged),
        "min_proxy_score_100": min(r["proxy_score_100"] for r in rows),
        "mean_proxy_score_100": float(np.mean([r["proxy_score_100"] for r in rows])),
    }

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== NO-GT REFINE SCORING ===")
    print("renders_dir         :", renders_dir)
    print("out_csv             :", out_csv)
    print("threshold_100       :", args.threshold_100)
    print("images              :", len(rows))
    print("flagged_images      :", len(flagged))
    print("proxy_score_mean    :", f"{summary['mean_proxy_score_100']:.2f}")
    print("proxy_score_min     :", f"{summary['min_proxy_score_100']:.2f}")
    print("\n=== TOP 10 ANH CAN REFINE ===")
    for idx, row in enumerate(rows[:10], start=1):
        print(
            f"{idx:>2}. {row['image']} | proxy={row['proxy_score_100']:.2f} | "
            f"global={row['global_score_100']:.2f} | tower={row['tower_score_100']:.2f} | "
            f"noise={row['noise_score_100']:.2f} | refine={row['refine_flag']}"
        )


if __name__ == "__main__":
    main()
