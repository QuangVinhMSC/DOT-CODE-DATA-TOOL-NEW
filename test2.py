import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = "orig.png"
BACKGROUND_PATH = "back.png"

MIN_SPACING = 1
MAX_SPACING = 25

DOT_RADIUS = None

INK_THRESHOLD = 0.03

PREVIEW_SCALE = 10


# ============================================================
# LOAD
# ============================================================

img = cv2.imread(IMAGE_PATH)
background_original = cv2.imread(BACKGROUND_PATH)

if img is None:
    raise FileNotFoundError(IMAGE_PATH)

if background_original is None:
    raise FileNotFoundError(BACKGROUND_PATH)

background = background_original.copy()


# ============================================================
# SELECT ROW
# ============================================================

roi = cv2.selectROI(
    "Select ONE dot row",
    img,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyWindow("Select ONE dot row")

x, y, w, h = map(int, roi)

if w <= 0 or h <= 0:
    raise RuntimeError("Khong co ROI.")

row = img[y:y+h, x:x+w].copy()


# ============================================================
# GRAYSCALE
# ============================================================

gray = cv2.cvtColor(
    row,
    cv2.COLOR_BGR2GRAY
).astype(np.float32)


# ============================================================
# ESTIMATE BACKGROUND
# ============================================================

flat = gray.flatten()

bright_count = max(
    int(len(flat) * 0.20),
    1
)

bright_pixels = np.partition(
    flat,
    len(flat) - bright_count
)[-bright_count:]

source_bg = float(
    np.median(bright_pixels)
)

print(
    f"Estimated source background: "
    f"{source_bg:.2f}"
)


# ============================================================
# CREATE INK MAP
#
# 0 = không có mực
# 1 = tối hoàn toàn
# ============================================================

ink = (
    source_bg - gray
) / max(source_bg, 1.0)

ink = np.clip(
    ink,
    0.0,
    1.0
)

ink[
    ink < INK_THRESHOLD
] = 0.0


# ============================================================
# HORIZONTAL PROFILE
# ============================================================

profile = np.sum(
    ink,
    axis=0
)

if profile.max() > 0:
    profile = profile / profile.max()


# ============================================================
# SMOOTH PROFILE
# ============================================================

kernel_size = 5

kernel = np.ones(
    kernel_size,
    dtype=np.float32
) / kernel_size

profile_smooth = np.convolve(
    profile,
    kernel,
    mode="same"
)


# ============================================================
# AUTOCORRELATION → SPACING
# ============================================================

p = profile_smooth - np.mean(profile_smooth)

autocorr = np.correlate(
    p,
    p,
    mode="full"
)

autocorr = autocorr[
    len(autocorr)//2:
]

max_s = min(
    MAX_SPACING,
    len(autocorr) - 1
)

min_s = min(
    MIN_SPACING,
    max_s
)

if max_s <= min_s:
    raise RuntimeError(
        "ROI qua nho de tim spacing."
    )

search = autocorr[
    min_s:max_s+1
]

spacing = (
    np.argmax(search)
    + min_s
)

print(
    f"Estimated spacing: {spacing} px"
)


# ============================================================
# FIND PHASE
# ============================================================

best_phase = 0
best_score = -1

for phase in range(spacing):

    xs = np.arange(
        phase,
        len(profile_smooth),
        spacing
    )

    if len(xs) < 2:
        continue

    score = np.sum(
        profile_smooth[xs]
    )

    if score > best_score:
        best_score = score
        best_phase = phase


print(
    f"Estimated phase: {best_phase}"
)


# ============================================================
# DOT RADIUS
# ============================================================

if DOT_RADIUS is None:

    radius = max(
        int(spacing * 0.65),
        2
    )

else:

    radius = DOT_RADIUS


patch_size = radius * 2 + 1


# ============================================================
# X CENTERS
# ============================================================

centers_x = np.arange(
    best_phase,
    w,
    spacing
)

valid_centers = []

for cx in centers_x:

    if (
        cx - radius >= 0
        and
        cx + radius < w
    ):

        valid_centers.append(
            int(cx)
        )


if len(valid_centers) < 2:
    raise RuntimeError(
        "Khong du dot de reconstruction."
    )


print(
    f"Valid dots: {len(valid_centers)}"
)


# ============================================================
# VERTICAL CENTER
# ============================================================

vertical_profile = np.sum(
    ink,
    axis=1
)

cy = int(
    np.argmax(vertical_profile)
)

print(
    f"Row center Y: {cy}"
)


# ============================================================
# EXTRACT REPEATED PATCHES
# ============================================================

patches = []

for cx in valid_centers:

    y1 = cy - radius
    y2 = cy + radius + 1

    x1 = cx - radius
    x2 = cx + radius + 1

    if (
        y1 < 0
        or y2 > h
        or x1 < 0
        or x2 > w
    ):
        continue

    patch = ink[
        y1:y2,
        x1:x2
    ].copy()

    if patch.shape != (
        patch_size,
        patch_size
    ):
        continue

    patches.append(
        patch
    )


if len(patches) < 2:
    raise RuntimeError(
        "Khong du patch."
    )


patches = np.array(
    patches,
    dtype=np.float32
)


# ============================================================
# MEAN PATCH
# ============================================================

mean_patch = np.mean(
    patches,
    axis=0
)


# ============================================================
# CENTRAL DOT REGION
# ============================================================

yy, xx = np.indices(
    mean_patch.shape,
    dtype=np.float32
)

center = radius

dx = xx - center

half_spacing = (
    spacing / 2.0
)

edge_width = max(
    spacing * 0.15,
    1.0
)

distance_to_boundary = (
    half_spacing
    - np.abs(dx)
)

soft_mask = np.clip(
    distance_to_boundary
    / edge_width,
    0.0,
    1.0
)

mean_dot = (
    mean_patch
    * soft_mask
)

mean_dot = np.clip(
    mean_dot,
    0.0,
    1.0
)

mean_dot[
    mean_dot < 0.01
] = 0.0


# ============================================================
# PREVIEW FUNCTION
# ============================================================

def ink_to_preview(ink_map):

    return np.clip(
        (1.0 - ink_map) * 255,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# DRAW DETECTED CENTERS
# ============================================================

marked = row.copy()

for cx in valid_centers:

    cv2.circle(
        marked,
        (cx, cy),
        2,
        (0, 0, 255),
        -1
    )

    cv2.circle(
        marked,
        (cx, cy),
        radius,
        (0, 255, 0),
        1
    )


# ============================================================
# PASTE DOT ON BACKGROUND
# ============================================================

def paste_dot(img, cx, cy, dot):

    r = dot.shape[0] // 2

    x1 = cx - r
    y1 = cy - r
    x2 = cx + r + 1
    y2 = cy + r + 1

    if (
        x1 < 0
        or y1 < 0
        or x2 > img.shape[1]
        or y2 > img.shape[0]
    ):
        print(
            "Vi tri qua sat bien."
        )
        return

    roi = img[
        y1:y2,
        x1:x2
    ].astype(np.float32)

    # dot là mức "ink"
    #
    # ink = 0  -> giữ background
    # ink = 1  -> đen
    #
    # Đây là kiểu multiply.
    ink3 = dot[:, :, None]

    result = (
        roi
        * (1.0 - ink3)
    )

    img[
        y1:y2,
        x1:x2
    ] = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# BACKGROUND MOUSE
# ============================================================

def background_mouse(
    event,
    x,
    y,
    flags,
    param
):

    global background

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    paste_dot(
        background,
        x,
        y,
        mean_dot
    )

    cv2.imshow(
        "BACKGROUND - CLICK TO DRAW",
        background
    )


# ============================================================
# SHOW PREVIEW
# ============================================================

mean_preview = ink_to_preview(
    mean_dot
)

raw_preview = ink_to_preview(
    mean_patch
)

mean_large = cv2.resize(
    mean_preview,
    None,
    fx=PREVIEW_SCALE,
    fy=PREVIEW_SCALE,
    interpolation=cv2.INTER_NEAREST
)

raw_large = cv2.resize(
    raw_preview,
    None,
    fx=PREVIEW_SCALE,
    fy=PREVIEW_SCALE,
    interpolation=cv2.INTER_NEAREST
)


cv2.namedWindow(
    "BACKGROUND - CLICK TO DRAW",
    cv2.WINDOW_NORMAL
)

cv2.imshow(
    "ROW",
    row
)

cv2.imshow(
    "DETECTED CENTERS",
    marked
)

cv2.imshow(
    "RAW MEAN PATCH",
    raw_large
)

cv2.imshow(
    "RECONSTRUCTED DOT",
    mean_large
)

cv2.imshow(
    "BACKGROUND - CLICK TO DRAW",
    background
)


cv2.setMouseCallback(
    "BACKGROUND - CLICK TO DRAW",
    background_mouse
)


# ============================================================
# INSTRUCTIONS
# ============================================================

print(
"""
============================================================

BACKGROUND:

  LEFT CLICK
      Ve reconstructed dot.

KEY:

  R
      Reset back.png

  S
      Save result.png

  D
      Save reconstructed_dot.png

  ESC
      Exit

============================================================
"""
)


# ============================================================
# LOOP
# ============================================================

while True:

    key = cv2.waitKey(20) & 0xFF

    if key == 27:
        break

    # ========================================================
    # RESET BACKGROUND
    # ========================================================

    elif key == ord("r"):

        background = (
            background_original.copy()
        )

        cv2.imshow(
            "BACKGROUND - CLICK TO DRAW",
            background
        )

        print(
            "Background reset."
        )

    # ========================================================
    # SAVE RESULT
    # ========================================================

    elif key == ord("s"):

        cv2.imwrite(
            "result.png",
            background
        )

        print(
            "Saved result.png"
        )

    # ========================================================
    # SAVE DOT
    # ========================================================

    elif key == ord("d"):

        cv2.imwrite(
            "reconstructed_dot.png",
            mean_preview
        )

        np.save(
            "reconstructed_dot.npy",
            mean_dot
        )

        print(
            "Saved reconstructed_dot.png"
        )


cv2.destroyAllWindows()