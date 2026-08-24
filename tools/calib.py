"""Calibrate job geometry against orig.png by rendering through the program
itself (no source changes) and scoring the alignment of the ink masks."""
import copy
import numpy as np, cv2
from dotgen.core.models import BackgroundSpec, Quad, CharSpec, LineSpec, CharFormat, DefectSpec
from dotgen.core.compose import compose

ORIG_BOX = (205, 385, 25, 580)   # y0, y1, x0, x1
THR = 110


def orig_mask():
    # Squeezed because this cannot rely on cv2.imread's own return shape:
    # importing ultralytics (which the test suite does, for its dataset
    # checker) replaces cv2.imread with an imdecode wrapper, and OpenCV 5's
    # imdecode returns HxWx1 for a grayscale read where imread returns HxW.
    # A trailing axis of one would broadcast wrongly against a rendered mask.
    g = np.squeeze(cv2.imread("orig.png", 0))
    y0, y1, x0, x1 = ORIG_BOX
    return (g[y0:y1, x0:x1] < THR).astype(np.float32)


def white_bg(w=1000, h=500):
    arr = np.full((h, w, 3), 183, np.uint8)
    return BackgroundSpec("white", (w, h), Quad([(2.0, 2.0), (w - 2.0, 2.0),
                                                 (w - 2.0, h - 2.0), (2.0, h - 2.0)]), arr)


def render_mask(job, seed=0):
    j = copy.copy(job)
    j.backgrounds = [white_bg()]
    j.defects = DefectSpec(0, 0.0, 0, 0.0, 0, 0.0, 0.0)
    c = compose(j, 0, np.random.default_rng(seed))
    m = (cv2.cvtColor(c.image, cv2.COLOR_BGR2GRAY) < THR).astype(np.float32)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def score(a, b):
    """Best IoU of binary masks a, b over all translations (FFT correlation)."""
    H = max(a.shape[0], b.shape[0]) * 2
    W = max(a.shape[1], b.shape[1]) * 2
    A = np.zeros((H, W), np.float32); A[:a.shape[0], :a.shape[1]] = a
    B = np.zeros((H, W), np.float32); B[:b.shape[0], :b.shape[1]] = b
    fa = np.fft.rfft2(A); fb = np.fft.rfft2(B)
    corr = np.fft.irfft2(fa * np.conj(fb), s=(H, W))
    idx = int(np.argmax(corr)); dy, dx = divmod(idx, W)
    inter = float(corr.max())
    union = float(a.sum() + b.sum() - inter)
    return inter / union if union > 0 else 0.0
