# USAGE
# python eye_drowsiness_video.py --input active.mp4 --output output.mp4 --ear-thresh 0.20




# python eye_drowsiness_video.py --input eyes_closed.mp4 --output out.mp4 --combined-thresh 0.6 --intensity region --show-metrics
# --intensity center
# python eye_drowsiness_video.py --input active.mp4 --output out.mp4 --ear --ear-thresh 0.20 --show-metrics







import imutils
import numpy as np
import argparse
import cv2
import mediapipe as mp
from collections import deque
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye_landmarks):
	"""Calculate EAR from 6 eye landmarks"""
	if len(eye_landmarks) < 6:
		return 0.0
	
	A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
	B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
	C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
	
	if C == 0:
		return 0.0
	
	ear = (A + B) / (2.0 * C)
	return ear


def eye_perimeter_ratio(eye_landmarks):
	"""
	Calculate ratio of actual perimeter to ideal perimeter.
	Closed eyes have smaller perimeter.
	"""
	if len(eye_landmarks) < 6:
		return 0.0
	
	# Calculate actual perimeter
	perimeter = 0
	for i in range(len(eye_landmarks)):
		next_i = (i + 1) % len(eye_landmarks)
		perimeter += dist.euclidean(eye_landmarks[i], eye_landmarks[next_i])
	
	# Calculate horizontal distance (baseline)
	horizontal = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
	
	if horizontal == 0:
		return 0.0
	
	# Normalize perimeter by horizontal distance
	return perimeter / horizontal


def eye_area_ratio(eye_landmarks):
	"""
	Calculate approximate area of eye polygon.
	Closed eyes have smaller area.
	"""
	if len(eye_landmarks) < 6:
		return 0.0
	
	# Use Shoelace formula for polygon area
	eye_np = np.array(eye_landmarks)
	x = eye_np[:, 0]
	y = eye_np[:, 1]
	
	area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
	
	# Normalize by horizontal distance squared
	horizontal = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
	
	if horizontal == 0:
		return 0.0
	
	return area / (horizontal ** 2)



def compute_eye_intensity(frame_gray, eye_landmarks, method="center", margin=5):
	"""
	Analyze pixel intensity in eye region.
	Closed eyes may have different thermal signature.
	"""
	if len(eye_landmarks) < 6:
		return 0.0
	
	eye_np = np.array(eye_landmarks, dtype=np.int32)
	
	if method == "center":
		# Compute center of the eye region from landmarks
		cx = int(np.mean(eye_np[:, 0]))
		cy = int(np.mean(eye_np[:, 1]))
		
		# Clamp center to frame bounds
		h, w = frame_gray.shape[:2]
		if w == 0 or h == 0:
			return 0.0
		cx = np.clip(cx, 0, w - 1)
		cy = np.clip(cy, 0, h - 1)
		
		# Use normalized intensity at the center point
		intensity = float(frame_gray[cy, cx]) / 255.0
		
		return intensity
	
	elif method == "region":
		# Get bounding box with margin around eye landmarks
		x_min = max(0, np.min(eye_np[:, 0]) - margin)
		x_max = min(frame_gray.shape[1], np.max(eye_np[:, 0]) + margin)
		y_min = max(0, np.min(eye_np[:, 1]) - margin)
		y_max = min(frame_gray.shape[0], np.max(eye_np[:, 1]) + margin)
		
		if x_max <= x_min or y_max <= y_min:
			return 0.0
		
		eye_region = frame_gray[y_min:y_max, x_min:x_max]
		
		if eye_region.size == 0:
			return 0.0
		
		# Calculate variance over region (open eyes have more variance)
		variance = np.var(eye_region.astype(float) / 255.0)
		
		return variance
	
	# Fallback to center method if unknown option
	cx = int(np.mean(eye_np[:, 0]))
	cy = int(np.mean(eye_np[:, 1]))
	h, w = frame_gray.shape[:2]
	if w == 0 or h == 0:
		return 0.0
	cx = np.clip(cx, 0, w - 1)
	cy = np.clip(cy, 0, h - 1)
	intensity = float(frame_gray[cy, cx]) / 255.0
	return intensity


def combined_drowsiness_score(ear, perimeter_ratio, area_ratio, intensity_var):
	"""
	Combine multiple metrics into single drowsiness score.
	Returns: score between 0 (alert) and 1 (drowsy)
	"""
	# Normalize each metric (lower values = more drowsy for EAR, area, perimeter, intensity)
	ear_score = max(0, 1 - (ear / 0.3))  # EAR typically 0.15-0.3
	perimeter_score = max(0, 1 - (perimeter_ratio / 5.0))  # Typical range 3-5
	area_score = max(0, 1 - (area_ratio / 0.5))  # Typical range 0.2-0.5
	intensity_score = max(0, 1 - (intensity_var / 0.1))  # Typical range 0.05-0.15
	
	# Weighted combination (EAR and area are most reliable)
	weights = [0.4, 0.2, 0.3, 0.1]  # [ear, perimeter, area, intensity]
	scores = [ear_score, perimeter_score, area_score, intensity_score]
	
	combined = sum(w * s for w, s in zip(weights, scores))
	
	return combined


# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
ap.add_argument("--ear-thresh", type=float, default=0.20, help="EAR threshold")
ap.add_argument("--combined-thresh", type=float, default=0.6, 
	help="Combined drowsiness score threshold (0-1, higher = more sensitive)")
ap.add_argument("--window-sec", type=float, default=3.0, help="temporal window in seconds")
ap.add_argument("--majority-frac", type=float, default=0.5, 
	help="fraction of frames that must be drowsy")
ap.add_argument("--show-metrics", action='store_true', 
	help="show all metrics on screen")
ap.add_argument("--ear", action='store_true', 
	help="use EAR-only drowsiness (similar to eye_drowsiness_video_2.py)")
ap.add_argument("--intensity", choices=["center", "region"], default="center", 
	help="intensity method: 'center' uses center pixel, 'region' uses variance over eye region")
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

# Eye landmark indices
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]

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

drowsiness_history = deque(maxlen=window_size)
majority_frac = args["majority_frac"]

frame_count = 0

print("[INFO] Processing video...")
if args["ear"]:
	print(f"[INFO] EAR-only mode, EAR threshold: {args['ear_thresh']:.2f}")
else:
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

			# Extract eye landmarks
			left_eye = [landmarks[i] for i in LEFT_EYE_IDXS]
			right_eye = [landmarks[i] for i in RIGHT_EYE_IDXS]

			# Calculate all metrics for both eyes
			left_ear = eye_aspect_ratio(left_eye)
			right_ear = eye_aspect_ratio(right_eye)
			avg_ear = (left_ear + right_ear) / 2.0

			left_perim = eye_perimeter_ratio(left_eye)
			right_perim = eye_perimeter_ratio(right_eye)
			avg_perim = (left_perim + right_perim) / 2.0

			left_area = eye_area_ratio(left_eye)
			right_area = eye_area_ratio(right_eye)
			avg_area = (left_area + right_area) / 2.0

			left_intensity = compute_eye_intensity(gray, left_eye, method=args["intensity"])
			right_intensity = compute_eye_intensity(gray, right_eye, method=args["intensity"])
			avg_intensity = (left_intensity + right_intensity) / 2.0
		
			# Optionally compute combined drowsiness score
			combined_score = None
			if not args["ear"]:
				combined_score = combined_drowsiness_score(
					avg_ear, avg_perim, avg_area, avg_intensity
				)

			# Draw eye contours
			left_eye_np = np.array(left_eye, dtype=np.int32)
			right_eye_np = np.array(right_eye, dtype=np.int32)
			cv2.polylines(frame_copy, [left_eye_np], True, (0, 255, 0), 2)
			cv2.polylines(frame_copy, [right_eye_np], True, (0, 255, 0), 2)

			# Display metrics
			y_pos = 30
			cv2.putText(frame_copy, f"EAR: {avg_ear:.3f}", 
				(10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			
			y_pos += 30
			if combined_score is not None:
				cv2.putText(frame_copy, f"Combined Score: {combined_score:.3f}", 
					(10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
			else:
				cv2.putText(frame_copy, f"Mode: EAR-only (thr={args['ear_thresh']:.2f})", 
					(10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

			if args["show_metrics"]:
				y_pos += 25
				cv2.putText(frame_copy, f"Perim: {avg_perim:.2f} Area: {avg_area:.3f} Int: {avg_intensity:.3f}", 
					(10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

			# Drowsiness decision based on selected mode
			if args["ear"]:
				frame_is_drowsy = 1 if avg_ear < args["ear_thresh"] else 0
			else:
				frame_is_drowsy = 1 if combined_score > args["combined_thresh"] else 0
			drowsiness_history.append(frame_is_drowsy)

			if len(drowsiness_history) > 0:
				drowsy_count = sum(drowsiness_history)
				is_drowsy = drowsy_count >= majority_frac * len(drowsiness_history)
			else:
				is_drowsy = False

			status_text = "DROWSY" if is_drowsy else "ALERT"
			status_color = (0, 0, 255) if is_drowsy else (0, 255, 0)
			
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
