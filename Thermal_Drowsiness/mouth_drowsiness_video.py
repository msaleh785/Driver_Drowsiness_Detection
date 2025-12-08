# USAGE
# python mouth_detection.py --input video.mp4 --output output.mp4 --mar-thresh 0.6

import imutils
import numpy as np
import argparse
import cv2
import mediapipe as mp
from collections import deque
from scipy.spatial import distance as dist


def mouth_aspect_ratio(mouth_landmarks):
	"""
	Calculate MAR (Mouth Aspect Ratio) from mouth landmarks.
	Higher values indicate open mouth.
	Uses the complete 20-point outer lip contour.
	"""
	if len(mouth_landmarks) < 40:
		return 0.0
	
	# Using indices from the complete outer lip contour
	# Vertical distances at key points (top to bottom)
	# Left side (index 2 to 18 in outer lip)
	A = dist.euclidean(mouth_landmarks[2], mouth_landmarks[18])  
	# Center-left (index 4 to 16)
	B = dist.euclidean(mouth_landmarks[4], mouth_landmarks[16])  
	# Center (index 5 to 15)
	C = dist.euclidean(mouth_landmarks[5], mouth_landmarks[15])  
	# Center-right (index 6 to 14)
	D = dist.euclidean(mouth_landmarks[6], mouth_landmarks[14])  
	# Right side (index 8 to 12)
	E = dist.euclidean(mouth_landmarks[8], mouth_landmarks[12])  
	
	# Horizontal distance (left corner to right corner)
	# Index 0 (left) to index 10 (right) in outer lip
	F = dist.euclidean(mouth_landmarks[0], mouth_landmarks[10])
	
	if F == 0:
		return 0.0
	
	# Average vertical distance divided by horizontal
	mar = (A + B + C + D + E) / (5.0 * F)
	return mar


def mouth_opening_ratio(mouth_landmarks):
	"""
	Calculate ratio of mouth height to width.
	Open mouth has larger ratio.
	Uses the complete outer lip contour.
	"""
	if len(mouth_landmarks) < 40:
		return 0.0
	
	# Height: vertical distance at center (index 5 to 15 in outer lip)
	height = dist.euclidean(mouth_landmarks[5], mouth_landmarks[15])
	
	# Width: horizontal distance (index 0 to 10 in outer lip)
	width = dist.euclidean(mouth_landmarks[0], mouth_landmarks[10])
	
	if width == 0:
		return 0.0
	
	return height / width


def mouth_perimeter_ratio(mouth_landmarks):
	"""
	Calculate ratio of outer perimeter to inner perimeter.
	Open mouth has different perimeter characteristics.
	"""
	if len(mouth_landmarks) < 40:
		return 0.0
	
	# Outer lip perimeter (first 20 points)
	outer_perimeter = 0
	for i in range(20):
		next_i = (i + 1) % 20
		outer_perimeter += dist.euclidean(mouth_landmarks[i], mouth_landmarks[next_i])
	
	# Inner lip perimeter (points 20-39)
	inner_perimeter = 0
	for i in range(20, 40):
		next_i = 20 + ((i - 20 + 1) % 20)
		inner_perimeter += dist.euclidean(mouth_landmarks[i], mouth_landmarks[next_i])
	
	# Width for normalization
	width = dist.euclidean(mouth_landmarks[0], mouth_landmarks[10])
	
	if width == 0:
		return 0.0
	
	# Return normalized perimeter difference
	return (outer_perimeter - inner_perimeter) / width


def mouth_area_ratio(mouth_landmarks):
	"""
	Calculate approximate area of mouth polygon.
	Open mouth has larger area.
	"""
	if len(mouth_landmarks) < 40:
		return 0.0
	
	# Use outer lip points for area calculation (first 20 points)
	outer_lips = mouth_landmarks[:20]
	mouth_np = np.array(outer_lips)
	x = mouth_np[:, 0]
	y = mouth_np[:, 1]
	
	# Shoelace formula
	area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
	
	# Normalize by width squared
	width = dist.euclidean(mouth_landmarks[0], mouth_landmarks[10])
	
	if width == 0:
		return 0.0
	
	return area / (width ** 2)


def combined_mouth_score(mar, opening_ratio, perimeter_ratio, area_ratio):
	"""
	Combine multiple metrics into single mouth opening score.
	Returns: score between 0 (closed) and 1 (open)
	"""
	# Normalize each metric (higher values = more open)
	# Stricter thresholds: closed mouth ~0.05-0.20, open mouth ~0.35+
	# Using a steeper curve to better separate closed from open
	mar_score = min(1.0, max(0, (mar - 0.25) / 0.30))  # Only count MAR above 0.25
	opening_score = min(1.0, max(0, (opening_ratio - 0.25) / 0.30))  # Only count above 0.25
	perimeter_score = min(1.0, max(0, (perimeter_ratio - 0.3) / 1.5))  # Baseline at 0.3
	area_score = min(1.0, max(0, (area_ratio - 0.15) / 1.0))  # Baseline at 0.15
	
	# Weighted combination (MAR and opening ratio are most reliable)
	weights = [0.45, 0.45, 0.05, 0.05]  # [mar, opening, perimeter, area]
	scores = [mar_score, opening_score, perimeter_score, area_score]
	
	combined = sum(w * s for w, s in zip(weights, scores))
	
	return combined


# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
ap.add_argument("--mar-thresh", type=float, default=0.30, help="MAR threshold")
ap.add_argument("--combined-thresh", type=float, default=0.45, 
	help="Combined mouth opening score threshold (0-1, higher = more sensitive)")
ap.add_argument("--window-sec", type=float, default=2.0, help="temporal window in seconds")
ap.add_argument("--majority-frac", type=float, default=0.5, 
	help="fraction of frames that must show open mouth")
ap.add_argument("--show-metrics", action='store_true', 
	help="show all metrics on screen")
args = vars(ap.parse_args())

# Initialize MediaPipe Face Mesh
print("[INFO] Loading MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
	max_num_faces=1,
	refine_landmarks=True,
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5
)

# Complete mouth landmark indices from MediaPipe Face Mesh
# Lips outer contour - complete loop
LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185
]

# Lips inner contour - complete loop  
LIPS_INNER = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191
]

# Combine for all mouth landmarks
MOUTH_IDXS = LIPS_OUTER + LIPS_INNER

# Video capture
vs = cv2.VideoCapture(args["input"])
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 960, 540)

writer = None
fps = vs.get(cv2.CAP_PROP_FPS)
if fps <= 0:
	fps = 25.0

window_size = int(fps * args["window_sec"])
if window_size < 1:
	window_size = 1

mouth_open_history = deque(maxlen=window_size)
majority_frac = args["majority_frac"]

frame_count = 0

print("[INFO] Processing video...")
print(f"[INFO] Combined threshold: {args['combined_thresh']:.2f} (adjust if needed)")

while True:
	grabbed, frame = vs.read()
	if not grabbed:
		break

	frame_count += 1
	frame = imutils.resize(frame, width=960)
	frame_copy = frame.copy()
	gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
	h, w, _ = frame_copy.shape

	frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
	results = face_mesh.process(frame_rgb)

	if results.multi_face_landmarks:
		for face_landmarks in results.multi_face_landmarks:
			# Extract landmarks
			landmarks = []
			for landmark in face_landmarks.landmark:
				x = int(landmark.x * w)
				y = int(landmark.y * h)
				landmarks.append((x, y))

			# Extract mouth landmarks
			mouth = [landmarks[i] for i in MOUTH_IDXS]

			# Calculate all metrics
			mar = mouth_aspect_ratio(mouth)
			opening_ratio = mouth_opening_ratio(mouth)
			perimeter_ratio = mouth_perimeter_ratio(mouth)
			area_ratio = mouth_area_ratio(mouth)

			# Compute combined mouth opening score
			combined_score = combined_mouth_score(
				mar, opening_ratio, perimeter_ratio, area_ratio
			)

			# Draw mouth contours
			mouth_outer = np.array(mouth[:20], dtype=np.int32)
			mouth_inner = np.array(mouth[20:], dtype=np.int32)
			cv2.polylines(frame_copy, [mouth_outer], True, (0, 255, 0), 2)
			cv2.polylines(frame_copy, [mouth_inner], True, (255, 0, 0), 2)

			# Display metrics
			y_pos = 30
			cv2.putText(frame_copy, f"MAR: {mar:.3f}", 
				(10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			
			y_pos += 30
			cv2.putText(frame_copy, f"Combined Score: {combined_score:.3f}", 
				(10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

			if args["show_metrics"]:
				y_pos += 25
				cv2.putText(frame_copy, f"Open: {opening_ratio:.2f} Perim: {perimeter_ratio:.2f} Area: {area_ratio:.2f}", 
					(10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

			# Mouth open decision based on combined score
			frame_mouth_open = 1 if combined_score > args["combined_thresh"] else 0
			mouth_open_history.append(frame_mouth_open)

			if len(mouth_open_history) > 0:
				open_count = sum(mouth_open_history)
				is_mouth_open = open_count >= majority_frac * len(mouth_open_history)
			else:
				is_mouth_open = False

			status_text = "MOUTH OPEN" if is_mouth_open else "MOUTH CLOSED"
			status_color = (0, 165, 255) if is_mouth_open else (0, 255, 0)
			
			y_pos += 30
			cv2.putText(frame_copy, f"STATUS: {status_text}", 
				(10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

	else:
		cv2.putText(frame_copy, "No face detected", 
			(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# Write video
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		writer = cv2.VideoWriter(args["output"], fourcc, fps,
			(frame_copy.shape[1], frame_copy.shape[0]), True)

	writer.write(frame_copy)
	cv2.imshow("Frame", frame_copy)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# Cleanup
cv2.destroyAllWindows()
vs.release()
if writer:
	writer.release()
face_mesh.close()

print(f"[INFO] Processed {frame_count} frames")
print("[INFO] Done!")