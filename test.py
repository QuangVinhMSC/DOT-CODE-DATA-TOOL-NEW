import cv2
import numpy as np

# =========================
# 1. Chọn ảnh
# =========================
image_path = r"C:\Users\dangq\OneDrive\Desktop\background\orig.png"

img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Không mở được ảnh: {image_path}")

# =========================
# 2. Chọn vùng ROI bằng chuột
# =========================
x, y, w, h = cv2.selectROI(
    "Chon vung can do",
    img,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

if w == 0 or h == 0:
    print("Không có vùng nào được chọn.")
    exit()

roi = img[y:y+h, x:x+w]

# =========================
# 3. Chuyển sang grayscale
# =========================
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# =========================
# 4. Tính độ sáng
# =========================
mean_brightness = np.mean(gray)
min_brightness = np.min(gray)
max_brightness = np.max(gray)
std_brightness = np.std(gray)

print(f"ROI: x={x}, y={y}, w={w}, h={h}")
print(f"Độ sáng trung bình : {mean_brightness:.2f}")
print(f"Tối nhất           : {min_brightness}")
print(f"Sáng nhất          : {max_brightness}")
print(f"Độ lệch chuẩn      : {std_brightness:.2f}")

# =========================
# 5. Hiển thị ROI
# =========================
cv2.imshow("ROI", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()