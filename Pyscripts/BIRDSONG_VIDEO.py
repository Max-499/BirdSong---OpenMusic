import cv2
import numpy as np
import csv
import os

# --------------------------
# Parameters
# --------------------------
VIDEO_PATH = "/Users/maxkeune/Documents/GitHub/BirdSong---OpenMusic/Images/Flying Birds 4K Video - Free HD Stock Footage - No Copyright - Bird Animals Sky.mp4"
OUTPUT_CSV = "birds_video_normalized.csv"
OUTPUT_LISP = "birds_video.lisp"
OUTPUT_DEBUG_DIR = "video_debug_frames"
OUTPUT_DIR = "/Users/maxkeune/Documents/GitHub/BirdSong---OpenMusic/BirdWorkSpace/user"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DEBUG_DIR, exist_ok=True)

MERGE_DISTANCE = 20
MIN_AREA = 30
MAX_AREA = 500
FRAME_INTERVAL = 0.5  # seconds between frames

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
# Open video
# --------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval_count = int(fps * FRAME_INTERVAL)

all_csv_data = []
all_x_raw, all_y_raw = [], []
all_x_norm, all_y_midi = [], []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval_count == 0:
        height, width = frame.shape[:2]

        # --------------------------
        # Process frame
        # --------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        gray_clahe = cv2.normalize(gray_clahe, None, 0, 255, cv2.NORM_MINMAX)
        gray_clahe = cv2.convertScaleAbs(gray_clahe, alpha=1.5, beta=30)
        blurred = cv2.GaussianBlur(gray_clahe, (7,7), 0)

        # Adaptive threshold + morphology
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Contour detection
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

        points = merge_close_points(points, MERGE_DISTANCE)

        if points:
            # Sort birds left to right by X
            points = sorted(points, key=lambda p: p[0])

            xs = [cx for (cx, _) in points]
            x_min, x_max = min(xs), max(xs)
            if x_max - x_min < 1e-3:
                x_max += 1

            def norm_x(cx):
                return np.interp(cx, [x_min, x_max], [0,1])
            def norm_y(cy):
                cy_clipped = np.clip(cy, 0, height)
                return int(np.interp(cy_clipped, [0,height], [127,0]))


            # Append CSV data
            for (cx, cy) in points:
                all_csv_data.append([frame_idx/fps, norm_x(cx), norm_y(cy)])

            # Append Lisp dataset
            all_x_raw.extend([str(round(x)) for x,_ in points])
            all_y_raw.extend([str(round(y)) for _,y in points])
            all_x_norm.extend([str(round(norm_x(x),4)) for x,_ in points])
            all_y_midi.extend([str(norm_y(y)) for _,y in points])

            # Debug frame
            debug_frame = frame.copy()
            for i,(x,y) in enumerate(points):
                cv2.circle(debug_frame, (int(x), int(y)), 8, (0,0,255), 2)
                cv2.putText(debug_frame, str(i+1), (int(x)+10,int(y)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            debug_filename = os.path.join(OUTPUT_DEBUG_DIR, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(debug_filename, debug_frame)

    frame_idx += 1

cap.release()
print(f"Processed video: {VIDEO_PATH}")

# --------------------------
# Write CSV
# --------------------------
with open(os.path.join(OUTPUT_DIR, OUTPUT_CSV), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_s", "x_norm", "y_midi"])
    writer.writerows(all_csv_data)

print(f"Normalized CSV saved: {os.path.join(OUTPUT_DIR, OUTPUT_CSV)}")

# --------------------------
# Write Lisp file (single dataset)
# --------------------------
lisp_filename = os.path.join(OUTPUT_DIR, OUTPUT_LISP)
with open(lisp_filename, "w") as f:
    f.write(f";; Bird data from video: {os.path.basename(VIDEO_PATH)}\n")
    f.write(f"(defun bird-x-raw () '(" + " ".join(all_x_raw) + "))\n")
    f.write(f"(defun bird-y-raw () '(" + " ".join(all_y_raw) + "))\n")
    f.write(f"(defun bird-x () '(" + " ".join(all_x_norm) + "))\n")
    f.write(f"(defun bird-y () '(" + " ".join(all_y_midi) + "))\n")

print(f"Lisp file saved: {lisp_filename}")
