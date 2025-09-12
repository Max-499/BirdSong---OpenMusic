import cv2
import numpy as np
import csv
import os

# --------------------------
# Parameters
# --------------------------
IMAGE_PATH = "/Users/maxkeune/Documents/GitHub/BirdSong---OpenMusic/Images/Birds5.jpg"
OUTPUT_CSV = "birds_normalized_fullrange.csv"
OUTPUT_DEBUG = "birds_debug.jpg"

OUTPUT_DIR = "/Users/maxkeune/Documents/GitHub/BirdSong---OpenMusic/BirdWorkSpace/user"  #where the files will be saved, ideally OM user folder 
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
lisp_filename = os.path.join(OUTPUT_DIR, f"{image_name}_birds.lisp")

MERGE_DISTANCE = 250
MIN_AREA = 500
MAX_AREA = 10000

# --------------------------
# Helper: merge close points
# --------------------------
def merge_close_points(points, min_dist=MERGE_DISTANCE):
    merged = []
    for p in points:
        if not merged:
            merged.append(p)
        else:
            dists = [np.linalg.norm(np.array(p) - np.array(m)) for m in merged]
            if min(dists) < min_dist:
                idx = np.argmin(dists)
                merged[idx] = (
                    (merged[idx][0] + p[0]) / 2,
                    (merged[idx][1] + p[1]) / 2
                )
            else:
                merged.append(p)
    return merged

# --------------------------
# 1. Load and process image
# --------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load {IMAGE_PATH}")

height, width = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_clahe = clahe.apply(gray)

blurred = cv2.GaussianBlur(gray_clahe, (11, 11), 0)

# --------------------------
# 2. Detect birds 
# --------------------------
# Enhance contrast & brightness
gray_clahe = cv2.normalize(gray_clahe, None, 0, 255, cv2.NORM_MINMAX)
gray_clahe = cv2.convertScaleAbs(gray_clahe, alpha=1.5, beta=30)

# blur to reduce noise
blurred = cv2.GaussianBlur(gray_clahe, (7, 7), 0)

# Adaptive threshold to handle dark images
thresh = cv2.adaptiveThreshold(blurred, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# remove noise
kernel = np.ones((3,3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

points = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if MIN_AREA <= area <= MAX_AREA:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            points.append((cx, cy))

# Merge close points 
points = merge_close_points(points, MERGE_DISTANCE)

print("Birds detected after merging:", len(points))
if not points:
    raise RuntimeError("No birds detected. Check your image or thresholds.")

# --------------------------
# 3. sort birds
# --------------------------
points = sorted(points, key=lambda p: p[0])

# --------------------------
# 4. make bird-image (with indicators) 
# --------------------------
debug_img = img.copy()
for i, (x, y) in enumerate(points):
    cv2.circle(debug_img, (int(x), int(y)), 8, (0, 0, 255), 2)
    cv2.putText(debug_img, str(i+1), (int(x)+10, int(y)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.imwrite(OUTPUT_DEBUG, debug_img)
print(f"Debug image saved: {OUTPUT_DEBUG}")

# --------------------------
# 5. Normalize positions
# --------------------------
xs = [cx for (cx, _) in points]

x_min, x_max = min(xs), max(xs)

# Avoid division by zero
if x_max - x_min < 1e-3:
    x_max += 1

def norm_x(cx):
    return np.interp(cx, [x_min, x_max], [0, 1])

def norm_y(cy):
    cy_clipped = np.clip(cy, 0, height)
    return int(np.interp(cy_clipped, [0, height], [127, 0]))


# --------------------------
# 6. Prepare CSV
# --------------------------
csv_data = []
for (cx, cy) in points:
    csv_data.append([norm_x(cx), norm_y(cy)])

# --------------------------
# 7. Write CSV
# --------------------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x_norm", "y_midi"])
    writer.writerows(csv_data)

print(f"Normalized CSV saved: {OUTPUT_CSV}")


# Export as Lisp file (for OpenMusic)
# xs_raw / ys_raw from original points in pixels
xs_raw  = [str(round(x)) for x, _ in points]
ys_raw  = [str(round(y)) for _, y in points]

# xs_norm 0..1 and ys_midi 0..127
xs_norm = [str(round(x / width, 4)) for x, _ in points]
ys_midi = [str(int(round((1 - (y / height)) * 127))) for _, y in points]

with open(lisp_filename, "w") as f:
    f.write(f";; Bird data accessors for {image_name}\n")
    # raw pixel data
    f.write(f"(defun bird-x-raw-{image_name} () '(" + " ".join(xs_raw) + "))\n")
    f.write(f"(defun bird-y-raw-{image_name} () '(" + " ".join(ys_raw) + "))\n")
    # normalized / MIDI data
    f.write(f"(defun bird-x-{image_name} () '(" + " ".join(xs_norm) + "))\n")
    f.write(f"(defun bird-y-{image_name} () '(" + " ".join(ys_midi) + "))\n")


print(f"Wrote {lisp_filename}")


