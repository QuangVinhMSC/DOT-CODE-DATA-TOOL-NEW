"""Collect isolated printed dots from orig.png through the program's own
Tab-1 extraction path, and fit the PCA dot model with them."""
import cv2, numpy as np
from dotgen.core.models import ROI
from dotgen.core.dot_extract import ExtractConfig, extract_dot_ex
from dotgen.core.dot_pca import build_pca_model, dot_params

TEXT_BOX = (205, 385, 25, 580)   # y0, y1, x0, x1


def isolated_dots(gray, thr=150, min_area=45, max_area=200, clearance=6):
    y0, y1, x0, x1 = TEXT_BOX
    roi = gray[y0:y1, x0:x1]
    binm = (roi < thr).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(binm, 8)
    keep = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (min_area <= a <= max_area and 6 <= w <= 18 and 6 <= h <= 18):
            continue
        # roundness: the component should fill most of its bbox
        if a < 0.55 * w * h:
            continue
        keep.append((i, x, y, w, h, cent[i]))
    out = []
    for i, x, y, w, h, c in keep:
        # clearance: no other labelled ink within `clearance` px of the bbox
        xa, ya = max(0, x - clearance), max(0, y - clearance)
        xb, yb = min(lab.shape[1], x + w + clearance), min(lab.shape[0], y + h + clearance)
        patch = lab[ya:yb, xa:xb]
        others = set(np.unique(patch)) - {0, i}
        if others:
            continue
        out.append((float(c[0]) + x0, float(c[1]) + y0, w, h))
    return out


def circle_roi(shape, cx, cy, r):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(r), 255, -1)
    x, y = int(round(cx)) - r, int(round(cy)) - r
    return ROI("circle", mask, (x, y, 2 * r + 1, 2 * r + 1), (cx, cy))


def collect(path="orig.png", patch_radius=7, roi_r=11, thr=150):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cfg = ExtractConfig(patch_radius=patch_radius, threshold=thr)
    samples, reasons = [], {}
    for cx, cy, w, h in isolated_dots(gray, thr=thr):
        s, why = extract_dot_ex(img, circle_roi(img.shape, cx, cy, roi_r), cfg)
        if s is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        samples.append(s)
    return samples, reasons


def collect_peaks(path="orig.png", patch_radius=7, roi_r=8, thr=150, min_dist=5):
    """Sample every detected dot centre with a small circular ROI."""
    import dotpeaks
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cfg = ExtractConfig(patch_radius=patch_radius, threshold=thr)
    samples, reasons = [], {}
    for cx, cy, _ in dotpeaks.peaks(gray, thr=thr, min_dist=min_dist):
        s, why = extract_dot_ex(img, circle_roi(img.shape, cx, cy, roi_r), cfg)
        if s is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        samples.append(s)
    return samples, reasons
