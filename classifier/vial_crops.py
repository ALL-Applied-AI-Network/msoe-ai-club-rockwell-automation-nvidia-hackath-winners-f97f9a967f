"""Position crops shared by make_crops.py (training) and predict.py (inference), so both cut identically.

A position is a dict like the sdg_vials.py labels: G positions have aperture_px and radius_px; S positions
have axis_px and diam_px (or, in older labels, only bbox_px).
"""
import math

import cv2
import numpy as np

PAD = 96            # border added around the frame so crops near the edge stay full size
G_SCALE = 5.0       # G crop side as a multiple of the aperture radius
S_SCALE = 1.15      # S crop side as a multiple of the vial length
S_BAND = 0.8        # S target band half-height as a multiple of the vial diameter
S_DIM = 0.35        # brightness outside the S target band


def pad(img):
    return cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT)


def g_crop(padded, info, size, g_scale=G_SCALE):
    """Square crop around a G aperture; None if the position has no aperture."""
    if not info.get("aperture_px") or not info.get("radius_px"):
        return None
    h = int(round(info["radius_px"] * g_scale / 2))
    cx, cy = (int(round(v)) + PAD for v in info["aperture_px"])
    crop = padded[cy - h:cy + h, cx - h:cx + h]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if 2 * h > size else cv2.INTER_CUBIC)


def s_axis(info):
    """(u0, v0, u1, v1, diam) of a lying vial, from axis_px or, for older labels, a horizontal axis through bbox_px."""
    if info.get("axis_px") and info.get("diam_px"):
        (u0, v0), (u1, v1) = info["axis_px"]
        return u0, v0, u1, v1, info["diam_px"]
    if info.get("bbox_px"):
        x0, y0, x1, y1 = info["bbox_px"]
        return x0, (y0 + y1) / 2, x1, (y0 + y1) / 2, y1 - y0
    return None


def s_crop(padded, info, size, s_scale=S_SCALE, s_band=S_BAND, s_dim=S_DIM):
    """Crop along a lying vial's axis, rotated horizontal (closed end left), outside its band dimmed."""
    ax = s_axis(info)
    if ax is None:
        return None
    u0, v0, u1, v1, diam = ax
    length = math.hypot(u1 - u0, v1 - v0)
    if length < 4:
        return None
    side = length * s_scale
    ang = math.degrees(math.atan2(v1 - v0, u1 - u0))
    cx, cy = (u0 + u1) / 2 + PAD, (v0 + v1) / 2 + PAD
    M = cv2.getRotationMatrix2D((cx, cy), ang, size / side)
    M[:, 2] += [size / 2 - cx, size / 2 - cy]
    crop = cv2.warpAffine(padded, M, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    band = diam * s_band * size / side
    yy = np.abs(np.arange(size) - size / 2)[:, None]
    return (crop * np.where(yy <= band, 1.0, s_dim)[..., None]).astype(np.uint8)


def crop_position(padded, pos, info, size):
    return g_crop(padded, info, size) if pos.startswith("G") else s_crop(padded, info, size)
