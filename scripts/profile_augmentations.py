"""
Profile augmentation routines using cProfile.

This script benchmarks:
- Albumentations transform pipeline (as configured in DataProcessor)
- Custom lighting_augmentation
- Primitive effect generators: lines, gradient, circular, perlin

Usage examples:
  PYTHONPATH=. python -X faulthandler -m cProfile -s tottime scripts/profile_augmentations.py --iters 25 --h 720 --w 1280
  PYTHONPATH=. python scripts/profile_augmentations.py --iters 10 --fast
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np

# Ensure project root on sys.path if running directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from data_generation.augment_images import (
	DataProcessor,
	lighting_augmentation,
	lines,
	gradient,
	circular,
	perlin,
)


def run_pipeline(dp: DataProcessor, img: np.ndarray, iters: int, seed: int | None = None) -> float:
	if seed is not None:
		np.random.seed(seed)
	h, w = img.shape[:2]
	accum = 0.0
	base = img
	for _ in range(iters):
		# Albumentations pipeline
		out = dp.albumentations_transform(image=base)["image"]
		# Custom lighting augmentation
		out = lighting_augmentation(out)
		# Primitive backgrounds/effects
		bg1 = lines(w, h)
		bg2 = gradient(w, h)
		bg3 = circular(w, h)
		bg4 = perlin(w, h)
		# Combine trivially so work isn't optimized away
		if out.ndim == 2:
			out3 = np.repeat(out[:, :, None], 3, axis=2)
		else:
			out3 = out
		combo = out3.astype(np.float32)
		combo *= bg1.reshape(h, w, 1).astype(np.float32)
		combo *= np.clip(bg2.reshape(h, w, 1) * 1.2 + 0.1, 0, 1)
		combo *= np.clip(bg3.reshape(h, w, 1) * 0.8 + 0.2, 0, 1)
		combo *= np.clip(bg4.reshape(h, w, 1) * 1.0, 0, 1)
		accum += float(combo.mean())
		base = out3  # carry to next iter
	return accum


def make_image(h: int, w: int, channels: int, pattern: str = "noise", seed: int | None = 123) -> np.ndarray:
	if seed is not None:
		rng = np.random.default_rng(seed)
	else:
		rng = np.random.default_rng()
	if pattern == "noise":
		img = (rng.random((h, w, channels)) * 255).astype(np.uint8)
	elif pattern == "grad":
		y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
		x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
		base = (0.7 * y + 0.3 * x)
		img = np.stack([base, base, base], axis=-1).astype(np.uint8)
		if channels == 1:
			img = img[..., :1]
	else:
		img = np.full((h, w, channels), 127, dtype=np.uint8)
	return img


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="cProfile for augmentation pipeline")
	parser.add_argument("--iters", type=int, default=20, help="Number of iterations to run")
	parser.add_argument("--h", type=int, default=720, help="Image height")
	parser.add_argument("--w", type=int, default=1280, help="Image width")
	parser.add_argument("--channels", type=int, default=3, choices=[1, 3], help="Number of channels")
	parser.add_argument("--fast", action="store_true", help="Use fast-mode transforms")
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args(argv)

	# Initialize processor (no dataset required)
	dp = DataProcessor(data_folders=[], out_dir=os.getcwd(), fast_mode=args.fast)

	# Synthetic image
	img = make_image(args.h, args.w, args.channels, pattern="noise", seed=args.seed)

	t0 = time.perf_counter()
	s = run_pipeline(dp, img, iters=args.iters, seed=args.seed)
	dt = time.perf_counter() - t0
	px = args.h * args.w
	print(
		f"done: iters={args.iters} size={args.w}x{args.h} ch={args.channels} fast={args.fast} "
		f"time={dt:.3f}s per_iter={dt/args.iters:.4f}s MPix/s={(px*args.iters/1e6)/dt:.2f} checksum={s:.3f}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

