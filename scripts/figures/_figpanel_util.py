"""Shared helpers for supplement figure-panel composition (trim / resize).

Figure scripts import this from scripts/figures/ (allowed: same folder). No
analysis is run here; these are pure image-layout utilities.
"""
import numpy as np
from PIL import Image


def load(p) -> Image.Image:
    return Image.open(p).convert("RGB")


def trim(img: Image.Image, thresh: int = 248, pad: int = 8) -> Image.Image:
    a = np.asarray(img.convert("RGB"))
    m = np.any(a < thresh, axis=2)
    if not m.any():
        return img
    rs = np.where(m.any(axis=1))[0]; cs = np.where(m.any(axis=0))[0]
    h, w = a.shape[:2]
    return img.crop((max(0, cs[0] - pad), max(0, rs[0] - pad),
                     min(w, cs[-1] + pad + 1), min(h, rs[-1] + pad + 1)))


def to_w(im: Image.Image, w: int) -> Image.Image:
    return im.resize((int(w), max(1, round(im.height * w / im.width))), Image.LANCZOS)
