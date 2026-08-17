import cv2
import numpy as np
import random

# ============================================================
# CONFIG
# ============================================================

SOURCE_IMAGE = "orig.png"
TARGET_IMAGE = "back.png"

# Vùng crop quanh vị trí click.
# Ví dụ 20 => patch 41x41
PATCH_RADIUS = 7


# Dot tối trên nền sáng:
# pixel < THRESHOLD được xem là ứng viên thuộc dot
THRESHOLD = 150

# Component tối thiểu
MIN_COMPONENT_AREA = 8

# Sau khi threshold tìm được core của dot,
# dilate thêm để giữ gradient / transition quanh mép.
EDGE_MARGIN = 5

# Gaussian blur cho support mask
# dùng số lẻ: 1 = không blur, 3,5,7...
SUPPORT_BLUR = 5

# PCA
# Số principal components tối đa sử dụng
MAX_PCA_COMPONENTS = 20

# Giới hạn random PCA theo số sigma
PCA_SIGMA_LIMIT = 2.5

# Khi ít sample, có thể giảm variation để an toàn.
PCA_VARIATION_SCALE = 1.0

# Hiển thị preview phóng lớn
PREVIEW_SCALE = 6


# ============================================================
# LOAD
# ============================================================

source_original = cv2.imread(SOURCE_IMAGE)
target_original = cv2.imread(TARGET_IMAGE)

if source_original is None:
    raise FileNotFoundError(SOURCE_IMAGE)

if target_original is None:
    raise FileNotFoundError(TARGET_IMAGE)

source_display = source_original.copy()
target = target_original.copy()

PATCH_SIZE = PATCH_RADIUS * 2 + 1

# Mỗi sample sẽ là ink map PATCH_SIZE x PATCH_SIZE
samples = []

# Lưu thêm thông tin để debug
sample_infos = []


# ============================================================
# HELPER
# ============================================================

def shift_image(img, dx, dy, interpolation=cv2.INTER_LINEAR):
    """
    Dịch ảnh mà không wrap-around.
    """

    h, w = img.shape[:2]

    M = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def estimate_background(gray):
    """
    Ước lượng intensity nền bằng các pixel ở viền patch.
    """

    border = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])

    # Median chống outlier tốt hơn mean
    return float(np.median(border))


# ============================================================
# EXTRACT DOT
# ============================================================

def extract_dot(img, click_x, click_y):

    r = PATCH_RADIUS

    x1 = click_x - r
    y1 = click_y - r
    x2 = click_x + r + 1
    y2 = click_y + r + 1

    if (
        x1 < 0 or
        y1 < 0 or
        x2 > img.shape[1] or
        y2 > img.shape[0]
    ):
        print("Click qua gan bien.")
        return None

    patch = img[y1:y2, x1:x2].copy()

    gray = cv2.cvtColor(
        patch,
        cv2.COLOR_BGR2GRAY
    )

    # ========================================================
    # 1. THRESHOLD
    # ========================================================

    _, binary = cv2.threshold(
        gray,
        THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV
    )

    # ========================================================
    # 2. CONNECTED COMPONENTS
    # ========================================================

    n_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )

    if n_labels <= 1:
        print("Khong tim thay component.")
        return None

    patch_center = np.array(
        [PATCH_RADIUS, PATCH_RADIUS],
        dtype=np.float32
    )

    best_label = None
    best_score = float("inf")

    for i in range(1, n_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_AREA:
            continue

        centroid = centroids[i]

        dist = np.linalg.norm(
            centroid - patch_center
        )

        # component gần click nhất
        if dist < best_score:
            best_score = dist
            best_label = i

    if best_label is None:
        print("Khong tim thay dot hop le.")
        return None

    core_mask = (
        labels == best_label
    ).astype(np.uint8) * 255

    # ========================================================
    # 3. CENTROID
    # ========================================================

    M = cv2.moments(core_mask)

    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # ========================================================
    # 4. TẠO SUPPORT MASK
    #
    # Threshold chỉ xác định "core".
    # Sau đó mở rộng ra để lấy gradient ở mép.
    # ========================================================

    kernel_size = EDGE_MARGIN * 2 + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    support = cv2.dilate(
        core_mask,
        kernel
    )

    support_f = (
        support.astype(np.float32) / 255.0
    )

    if SUPPORT_BLUR > 1:

        k = SUPPORT_BLUR

        if k % 2 == 0:
            k += 1

        support_f = cv2.GaussianBlur(
            support_f,
            (k, k),
            0
        )

        support_f = np.clip(
            support_f,
            0.0,
            1.0
        )

    # ========================================================
    # 5. ESTIMATE BACKGROUND
    # ========================================================

    bg = estimate_background(gray)

    if bg < 1:
        print("Background estimate khong hop le.")
        return None

    # ========================================================
    # 6. INK MAP
    #
    # 0 = không có mực
    # 1 = cực tối so với background
    #
    # Ví dụ:
    # background = 210
    # dot pixel  = 42
    #
    # darkness = (210-42)/210 = 0.80
    # ========================================================

    gray_f = gray.astype(np.float32)

    ink = (
        bg - gray_f
    ) / max(bg, 1.0)

    ink = np.clip(
        ink,
        0.0,
        1.0
    )

    # chỉ giữ vùng liên quan đến dot
    ink *= support_f

    # ========================================================
    # 7. CENTER ALIGNMENT
    #
    # Đây là phần quan trọng khi học PCA.
    # Tất cả chấm phải có tâm gần giống nhau.
    # ========================================================

    desired_x = PATCH_RADIUS
    desired_y = PATCH_RADIUS

    dx = desired_x - cx
    dy = desired_y - cy

    ink_aligned = shift_image(
        ink,
        dx,
        dy,
        cv2.INTER_LINEAR
    )

    core_aligned = shift_image(
        core_mask,
        dx,
        dy,
        cv2.INTER_NEAREST
    )

    # ========================================================
    # DEBUG PREVIEW
    # ========================================================

    preview = (
        (1.0 - ink_aligned) * 255
    ).astype(np.uint8)

    return {
        "ink": ink_aligned.astype(np.float32),
        "core_mask": core_aligned,
        "gray": gray,
        "binary": binary,
        "background": bg,
        "centroid": (cx, cy),
        "shift": (dx, dy),
        "preview": preview
    }


# ============================================================
# PCA MODEL
# ============================================================

def build_pca_model():

    if len(samples) == 0:
        return None

    # N x pixels
    X = np.array(
        [s.flatten() for s in samples],
        dtype=np.float32
    )

    mean = np.mean(
        X,
        axis=0
    )

    # Chỉ 1 sample:
    # không có variation
    if len(samples) == 1:

        return {
            "mean": mean,
            "components": np.empty(
                (0, X.shape[1]),
                dtype=np.float32
            ),
            "score_std": np.empty(
                0,
                dtype=np.float32
            ),
            "scores": np.empty(
                (1, 0),
                dtype=np.float32
            )
        }

    X_centered = X - mean

    # ========================================================
    # SVD
    #
    # X_centered = U S Vt
    #
    # Vt chứa principal directions.
    # ========================================================

    U, S, Vt = np.linalg.svd(
        X_centered,
        full_matrices=False
    )

    max_possible = min(
        len(samples) - 1,
        X.shape[1],
        MAX_PCA_COMPONENTS
    )

    # Loại những component gần như zero
    # để test click cùng 1 dot hoạt động đúng.
    eigen_strength = S[:max_possible]

    valid = eigen_strength > 1e-7

    components = Vt[:max_possible][valid]

    if len(components) == 0:

        return {
            "mean": mean,
            "components": np.empty(
                (0, X.shape[1]),
                dtype=np.float32
            ),
            "score_std": np.empty(
                0,
                dtype=np.float32
            ),
            "scores": np.empty(
                (len(samples), 0),
                dtype=np.float32
            )
        }

    # Project sample thật lên PCA space
    scores = (
        X_centered @ components.T
    )

    score_std = np.std(
        scores,
        axis=0,
        ddof=0
    )

    return {
        "mean": mean,
        "components": components,
        "score_std": score_std,
        "scores": scores
    }


# ============================================================
# GENERATE SYNTHETIC DOT
# ============================================================

def generate_pca_dot():

    model = build_pca_model()

    if model is None:
        print("Chua co sample.")
        return None

    mean = model["mean"]
    components = model["components"]
    score_std = model["score_std"]

    # Nếu tất cả sample giống hệt nhau:
    # output chính là mean.
    generated = mean.copy()

    if len(components) > 0:

        coeffs = []

        for std in score_std:

            if std < 1e-8:
                coeff = 0.0

            else:

                coeff = np.random.normal(
                    0.0,
                    std * PCA_VARIATION_SCALE
                )

                limit = (
                    PCA_SIGMA_LIMIT *
                    std *
                    PCA_VARIATION_SCALE
                )

                coeff = np.clip(
                    coeff,
                    -limit,
                    limit
                )

            coeffs.append(coeff)

        coeffs = np.asarray(
            coeffs,
            dtype=np.float32
        )

        generated += (
            coeffs @ components
        )

    generated = generated.reshape(
        PATCH_SIZE,
        PATCH_SIZE
    )

    generated = np.clip(
        generated,
        0.0,
        1.0
    )

    return generated


# ============================================================
# PASTE DOT
# ============================================================

def paste_dot(img, cx, cy, ink):

    r = PATCH_RADIUS

    x1 = cx - r
    y1 = cy - r
    x2 = cx + r + 1
    y2 = cy + r + 1

    if (
        x1 < 0 or
        y1 < 0 or
        x2 > img.shape[1] or
        y2 > img.shape[0]
    ):
        print("Vi tri paste qua gan bien.")
        return

    roi = img[
        y1:y2,
        x1:x2
    ].astype(np.float32)

    # ========================================================
    # MULTIPLICATIVE INK MODEL
    #
    # ink = 0:
    #   giữ nguyên target
    #
    # ink = 1:
    #   thành đen
    #
    # Ví dụ:
    #
    # target = 200
    # ink    = 0.7
    #
    # output = 200 * (1 - 0.7) = 60
    #
    # Vì vậy không mang background source sang target.
    # ========================================================

    ink3 = ink[:, :, None]

    result = (
        roi *
        (1.0 - ink3)
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
# STATISTICS
# ============================================================

def print_statistics():

    print()
    print("=" * 70)
    print(f"NUMBER OF DOT SAMPLES: {len(samples)}")
    print("=" * 70)

    if len(samples) == 0:
        return

    X = np.array(samples)

    print(
        f"Average max ink: "
        f"{np.mean(np.max(X, axis=(1,2))):.4f}"
    )

    print(
        f"Average mean ink: "
        f"{np.mean(np.mean(X, axis=(1,2))):.4f}"
    )

    # Effective area
    areas = []

    for s in samples:

        area = np.sum(
            s > 0.10
        )

        areas.append(area)

    print(
        f"Area mean: {np.mean(areas):.2f} px"
    )

    print(
        f"Area std : {np.std(areas):.2f} px"
    )

    model = build_pca_model()

    if model is not None:

        print(
            f"PCA components: "
            f"{len(model['components'])}"
        )

        if len(model["score_std"]) > 0:

            print("PCA score STD:")

            for i, v in enumerate(
                model["score_std"]
            ):

                print(
                    f"  PC{i+1:02d}: "
                    f"{v:.6f}"
                )

    print("=" * 70)


# ============================================================
# PREVIEW
# ============================================================

def show_preview(name, img):

    if img is None:
        return

    if img.dtype != np.uint8:

        img_show = np.clip(
            img * 255,
            0,
            255
        ).astype(np.uint8)

    else:
        img_show = img

    large = cv2.resize(
        img_show,
        None,
        fx=PREVIEW_SCALE,
        fy=PREVIEW_SCALE,
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow(
        name,
        large
    )


# ============================================================
# SOURCE MOUSE
# ============================================================

def source_mouse(event, x, y, flags, param):

    global source_display

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    result = extract_dot(
        source_original,
        x,
        y
    )

    if result is None:
        return

    samples.append(
        result["ink"]
    )

    sample_infos.append(
        result
    )

    # ========================================================
    # DRAW SAMPLE MARKER
    # ========================================================

    cv2.circle(
        source_display,
        (x, y),
        PATCH_RADIUS,
        (0, 0, 255),
        1
    )

    cv2.putText(
        source_display,
        str(len(samples)),
        (
            x - 10,
            y - PATCH_RADIUS - 4
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
        cv2.LINE_AA
    )

    cv2.imshow(
        "SOURCE",
        source_display
    )

    show_preview(
        "BINARY",
        result["binary"]
    )

    show_preview(
        "CORE MASK",
        result["core_mask"]
    )

    show_preview(
        "EXTRACTED DOT",
        result["preview"]
    )

    print()
    print(
        f"DOT #{len(samples)}"
    )

    print(
        f"background = "
        f"{result['background']:.2f}"
    )

    print(
        f"centroid = "
        f"({result['centroid'][0]:.2f}, "
        f"{result['centroid'][1]:.2f})"
    )

    print(
        f"align shift = "
        f"({result['shift'][0]:.2f}, "
        f"{result['shift'][1]:.2f})"
    )

    print_statistics()


# ============================================================
# TARGET MOUSE
# ============================================================

def target_mouse(event, x, y, flags, param):

    global target

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(samples) == 0:

        print(
            "Hay click SOURCE de lay dot truoc."
        )

        return

    ink = generate_pca_dot()

    if ink is None:
        return

    paste_dot(
        target,
        x,
        y,
        ink
    )

    cv2.imshow(
        "TARGET",
        target
    )

    # preview:
    # trắng = background
    # đen = dot
    preview = (
        (1.0 - ink) * 255
    ).astype(np.uint8)

    show_preview(
        "GENERATED DOT",
        preview
    )


# ============================================================
# WINDOWS
# ============================================================

cv2.namedWindow(
    "SOURCE",
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    "TARGET",
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    "BINARY",
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    "CORE MASK",
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    "EXTRACTED DOT",
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    "GENERATED DOT",
    cv2.WINDOW_NORMAL
)


cv2.imshow(
    "SOURCE",
    source_display
)

cv2.imshow(
    "TARGET",
    target
)


cv2.setMouseCallback(
    "SOURCE",
    source_mouse
)

cv2.setMouseCallback(
    "TARGET",
    target_mouse
)


# ============================================================
# INSTRUCTIONS
# ============================================================

print(
"""
============================================================

SOURCE:
  Left click = lay dot mau.

TARGET:
  Left click = generate dot tu PCA.

KEYS:

  S = save result.png

  R = reset TARGET

  C = clear toan bo samples

  P = print PCA statistics

  M = show mean dot

  G = generate preview dot

  ESC = exit

============================================================
"""
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    key = cv2.waitKey(20) & 0xFF

    # --------------------------------------------------------
    # ESC
    # --------------------------------------------------------

    if key == 27:
        break

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    elif key == ord("s"):

        cv2.imwrite(
            "result.png",
            target
        )

        print(
            "Saved: result.png"
        )

    # --------------------------------------------------------
    # RESET TARGET
    # --------------------------------------------------------

    elif key == ord("r"):

        target = target_original.copy()

        cv2.imshow(
            "TARGET",
            target
        )

        print(
            "Target reset."
        )

    # --------------------------------------------------------
    # CLEAR SAMPLES
    # --------------------------------------------------------

    elif key == ord("c"):

        samples.clear()
        sample_infos.clear()

        source_display = (
            source_original.copy()
        )

        cv2.imshow(
            "SOURCE",
            source_display
        )

        print(
            "All samples cleared."
        )

    # --------------------------------------------------------
    # PRINT STATS
    # --------------------------------------------------------

    elif key == ord("p"):

        print_statistics()

    # --------------------------------------------------------
    # MEAN DOT
    # --------------------------------------------------------

    elif key == ord("m"):

        model = build_pca_model()

        if model is None:

            print(
                "No samples."
            )

            continue

        mean_dot = model[
            "mean"
        ].reshape(
            PATCH_SIZE,
            PATCH_SIZE
        )

        preview = (
            (1.0 - mean_dot) * 255
        ).astype(np.uint8)

        show_preview(
            "GENERATED DOT",
            preview
        )

        print(
            "Showing PCA mean dot."
        )

    # --------------------------------------------------------
    # GENERATE DOT PREVIEW
    # --------------------------------------------------------

    elif key == ord("g"):

        ink = generate_pca_dot()

        if ink is None:
            continue

        preview = (
            (1.0 - ink) * 255
        ).astype(np.uint8)

        show_preview(
            "GENERATED DOT",
            preview
        )


cv2.destroyAllWindows()