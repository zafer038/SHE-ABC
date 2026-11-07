from __future__ import annotations

from pathlib import Path

import numpy as np

CLASSES = ["impervious_surfaces", "building", "low_vegetation", "tree", "car", "clutter"]
COLOURS = {(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2, (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5}
IGNORE = 255


def _read(path: Path) -> np.ndarray:
    import tifffile
    return tifffile.imread(path)


def load_patches(data_dir: str | Path):
    d = Path(data_dir)
    if not (d / "Images").is_dir():
        raise FileNotFoundError(f"{d / 'Images'} not found (copy the Potsdam 'patches' folder there)")
    paths = sorted((d / "Images").glob("Image_*.tif"), key=lambda p: int(p.stem.split("_")[1]))
    images, labels, unknown = [], [], 0
    lut = {r*65536 + g*256 + b: c for (r, g, b), c in COLOURS.items()}
    for p in paths:
        idx = p.stem.split("_")[1]
        img = _read(p)
        lab = _read(d / "Labels" / f"Label_{idx}.tif")
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img, 0, -1)
        img = img[..., :3]
        if img.dtype != np.uint8:
            img = np.clip(img.astype(float)/img.max()*255, 0, 255).astype(np.uint8)
        if lab.ndim == 3 and lab.shape[0] in (3, 4) and lab.shape[-1] not in (3, 4):
            lab = np.moveaxis(lab, 0, -1)
        code = lab[..., 0].astype(np.int64)*65536 + lab[..., 1].astype(np.int64)*256 + lab[..., 2].astype(np.int64)
        cls = np.full(code.shape, IGNORE, np.uint8)
        for value, c in lut.items():
            cls[code == value] = c
        unknown += int((cls == IGNORE).sum())
        images.append(img)
        labels.append(cls)
    images, labels = np.stack(images), np.stack(labels)
    return images, labels, {"names": [q.stem for q in paths], "patches": len(paths), "unknown_label_pixels": unknown,
                            "unknown_fraction": unknown/labels.size,
                            "class_pixels": np.bincount(labels[labels != IGNORE].ravel(), minlength=6).tolist()}


def split(n: int, fractions=(0.70, 0.15, 0.15), seed: int = 2026):
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(round(fractions[0]*n)), int(round((fractions[0] + fractions[1])*n))
    return perm[:a], perm[a:b], perm[b:]


def tile_of(name: str, patches_per_tile: int = 400) -> int:
    return int(name.split("_")[1])//patches_per_tile


def tile_split(names, tiles=(4, 1, 1), seed: int = 2026, patches_per_tile: int = 400):
    tile = np.array([tile_of(n, patches_per_tile) for n in names])
    ids = np.unique(tile)
    if len(ids) != sum(tiles):
        raise SystemExit(f"expected {sum(tiles)} tiles, found {len(ids)}")
    perm = ids[np.random.default_rng(seed).permutation(len(ids))]
    n_te, n_va = tiles[2], tiles[1]
    sets = {"test": perm[:n_te], "val": perm[n_te:n_te + n_va], "train": perm[n_te + n_va:]}
    idx = {k: np.flatnonzero(np.isin(tile, v)) for k, v in sets.items()}
    return idx["train"], idx["val"], idx["test"], {k: sorted(int(t) for t in v) for k, v in sets.items()}
