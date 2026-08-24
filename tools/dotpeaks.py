import cv2, numpy as np

TEXT_BOX = (205, 385, 25, 580)


def peaks(gray, thr=150, min_dist=5, disc_r=5):
    y0, y1, x0, x1 = TEXT_BOX
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    ink = np.clip(1.0 - roi / max(1.0, float(np.percentile(roi, 95))), 0, 1)
    k = 2 * disc_r + 1
    disc = np.zeros((k, k), np.float32)
    cv2.circle(disc, (disc_r, disc_r), disc_r, 1.0, -1)
    disc /= disc.sum()
    resp = cv2.filter2D(ink, -1, disc)
    dark = roi < thr
    md = 2 * min_dist + 1
    mx = cv2.dilate(resp, np.ones((md, md), np.uint8))
    hit = (resp >= mx - 1e-6) & dark & (resp > 0.30)
    ys, xs = np.nonzero(hit)
    # collapse plateaux
    pts = []
    used = np.zeros_like(hit)
    order = np.argsort(-resp[ys, xs])
    for i in order:
        y, x = ys[i], xs[i]
        if used[max(0, y - min_dist):y + min_dist + 1, max(0, x - min_dist):x + min_dist + 1].any():
            continue
        used[y, x] = True
        pts.append((float(x) + x0, float(y) + y0, float(resp[y, x])))
    return pts
