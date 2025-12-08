# USAGE
# From standalone_mouth_drowsiness directory
# python eye_mouth_combined_video.py --input "ddd.mp4" --output combined_out.mp4 --eye-combined-thresh 0.6 --eye-intensity center --mouth-combined-thresh 0.45 --show-metrics

import imutils
import numpy as np
import argparse
import cv2
import mediapipe as mp
from collections import deque
from scipy.spatial import distance as dist
import time


# --------------------
# Eye metrics (copied from eye_drowsiness_video.py)
# --------------------

def eye_aspect_ratio(eye_landmarks):
	"""Calculate EAR from 6 eye landmarks"""
	if len(eye_landmarks) < 6:
		return 0.0
	A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
	B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
	C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
	if C == 0:
		return 0.0
	return (A + B) / (2.0 * C)


def eye_perimeter_ratio(eye_landmarks):
	"""Perimeter / horizontal distance (same as eye_drowsiness_video)."""
	if len(eye_landmarks) < 6:
		return 0.0
	perimeter = 0.0
	for i in range(len(eye_landmarks)):
		next_i = (i + 1) % len(eye_landmarks)
		perimeter += dist.euclidean(eye_landmarks[i], eye_landmarks[next_i])
	horizontal = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
	if horizontal == 0:
		return 0.0
	return perimeter / horizontal


def eye_area_ratio(eye_landmarks):
	"""Normalized polygon area (same as eye_drowsiness_video)."""
	if len(eye_landmarks) < 6:
		return 0.0
	eye_np = np.array(eye_landmarks)
	x = eye_np[:, 0]
	y = eye_np[:, 1]
	area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
	horizontal = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
	if horizontal == 0:
		return 0.0
	return area / (horizontal ** 2)


def compute_eye_intensity(frame_gray, eye_landmarks, method="center", margin=5):
	"""Intensity metric, identical to eye_drowsiness_video.py."""
	if len(eye_landmarks) < 6:
		return 0.0
	
	eye_np = np.array(eye_landmarks, dtype=np.int32)
	
	if method == "center":
		cx = int(np.mean(eye_np[:, 0]))
		cy = int(np.mean(eye_np[:, 1]))
		h, w = frame_gray.shape[:2]
		if w == 0 or h == 0:
			return 0.0
		cx = np.clip(cx, 0, w - 1)
		cy = np.clip(cy, 0, h - 1)
		intensity = float(frame_gray[cy, cx]) / 255.0
		return intensity
	elif method == "region":
		x_min = max(0, np.min(eye_np[:, 0]) - margin)
		x_max = min(frame_gray.shape[1], np.max(eye_np[:, 0]) + margin)
		y_min = max(0, np.min(eye_np[:, 1]) - margin)
		y_max = min(frame_gray.shape[0], np.max(eye_np[:, 1]) + margin)
		if x_max <= x_min or y_max <= y_min:
			return 0.0
		eye_region = frame_gray[y_min:y_max, x_min:x_max]
		if eye_region.size == 0:
			return 0.0
		variance = np.var(eye_region.astype(float) / 255.0)
		return variance
	# fallback to center
	cx = int(np.mean(eye_np[:, 0]))
	cy = int(np.mean(eye_np[:, 1]))
	h, w = frame_gray.shape[:2]
	if w == 0 or h == 0:
		return 0.0
	cx = np.clip(cx, 0, w - 1)
	cy = np.clip(cy, 0, h - 1)
	return float(frame_gray[cy, cx]) / 255.0


def combined_eye_score(ear, perimeter_ratio, area_ratio, intensity_var):
	"""Same formula as combined_drowsiness_score in eye_drowsiness_video.py."""
	ear_score = max(0, 1 - (ear / 0.3))          # EAR typically 0.15–0.3
	perimeter_score = max(0, 1 - (perimeter_ratio / 5.0))  # range 3–5
	area_score = max(0, 1 - (area_ratio / 0.5))  # range 0.2–0.5
	intensity_score = max(0, 1 - (intensity_var / 0.1))    # range 0.05–0.15
	weights = [0.4, 0.2, 0.3, 0.1]
	scores = [ear_score, perimeter_score, area_score, intensity_score]
	return sum(w * s for w, s in zip(weights, scores))


# --------------------
# Mouth metrics (copied from mouth_drowsiness_video.py)
# --------------------

def mouth_aspect_ratio(mouth_landmarks):
	if len(mouth_landmarks) < 40:
		return 0.0
	A = dist.euclidean(mouth_landmarks[2], mouth_landmarks[18])
	B = dist.euclidean(mouth_landmarks[4], mouth_landmarks[16])
	C = dist.euclidean(mouth_landmarks[5], mouth_landmarks[15])
	D = dist.euclidean(mouth_landmarks[6], mouth_landmarks[14])
	E = dist.euclidean(mouth_landmarks[8], mouth_landmarks[12])
	F = dist.euclidean(mouth_landmarks[0], mouth_landmarks[10])
	if F == 0:
		return 0.0
	return (A + B + C + D + E) / (5.0 * F)


def mouth_opening_ratio(mouth_landmarks):
	if len(mouth_landmarks) < 40:
		return 0.0
	height = dist.euclidean(mouth_landmarks[5], mouth_landmarks[15])
	width = dist.euclidean(mouth_landmarks[0], mouth_landmarks[10])
	if width == 0:
		return 0.0
	return height / width


def mouth_perimeter_ratio(mouth_landmarks):
	if len(mouth_landmarks) < 40:
		return 0.0
	outer_perimeter = 0.0
	for i in range(20):
		next_i = (i + 1) % 20
		outer_perimeter += dist.euclidean(mouth_landmarks[i], mouth_landmarks[next_i])
	inner_perimeter = 0.0
	for i in range(20, 40):
		next_i = 20 + ((i - 20 + 1) % 20)
		inner_perimeter += dist.euclidean(mouth_landmarks[i], mouth_landmarks[next_i])
	width = dist.euclidean(mouth_landmarks[0], mouth_landmarks[10])
	if width == 0:
		return 0.0
	return (outer_perimeter - inner_perimeter) / width


def mouth_area_ratio(mouth_landmarks):
	if len(mouth_landmarks) < 40:
		return 0.0
	outer_lips = mouth_landmarks[:20]
	mouth_np = np.array(outer_lips)
	x = mouth_np[:, 0]
	y = mouth_np[:, 1]
	area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
	width = dist.euclidean(mouth_landmarks[0], mouth_landmarks[10])
	if width == 0:
		return 0.0
	return area / (width ** 2)


def combined_mouth_score(mar, opening_ratio, perimeter_ratio, area_ratio):
	mar_score = min(1.0, max(0, (mar - 0.25) / 0.30))
	opening_score = min(1.0, max(0, (opening_ratio - 0.25) / 0.30))
	perimeter_score = min(1.0, max(0, (perimeter_ratio - 0.3) / 1.5))
	area_score = min(1.0, max(0, (area_ratio - 0.15) / 1.0))
	weights = [0.45, 0.45, 0.05, 0.05]
	scores = [mar_score, opening_score, perimeter_score, area_score]
	return sum(w * s for w, s in zip(weights, scores))


# --------------------
# Argument parser
# --------------------

ap = argparse.ArgumentParser()

ap.add_argument("-v", "--input", required=True, help="path to input video or camera index (when --camera)")
ap.add_argument("-o", "--output", required=True, help="path to output video")

ap.add_argument("--camera", action='store_true',
			help="use live camera; when set, --input should be camera index (e.g., 0)")

# Eye (uses exact combined_eye_score logic)
ap.add_argument("--eye-combined-thresh", type=float, default=0.6,
			help="eye combined score threshold (same role as combined-thresh in eye_drowsiness_video)")
ap.add_argument("--eye-window-sec", type=float, default=3.0,
			help="eye temporal window in seconds (default 3.0)")
ap.add_argument("--eye-majority-frac", type=float, default=0.5,
			help="fraction of frames that must be drowsy in eye window")
ap.add_argument("--eye-intensity", choices=["center", "region"], default="region",
			help="eye intensity method: center or region (same as eye_drowsiness_video)")

# Mouth (uses exact combined_mouth_score logic)
ap.add_argument("--mouth-combined-thresh", type=float, default=0.45,
			help="mouth combined score threshold (same default as mouth_drowsiness_video)")
ap.add_argument("--mouth-window-sec", type=float, default=2.0,
			help="mouth temporal window in seconds (default 2.0)")
ap.add_argument("--mouth-majority-frac", type=float, default=0.5,
			help="fraction of frames that must show open mouth in mouth window")

# Final combination weights and thresholds
ap.add_argument("--eye-weight", type=float, default=0.7,
			help="weight of eye drowsiness in final combined score")
ap.add_argument("--mouth-weight", type=float, default=0.3,
			help="weight of mouth opening in final combined score")
ap.add_argument("--medium-thresh", type=float, default=0.4,
			help="combined score threshold for MEDIUM drowsiness")
ap.add_argument("--high-thresh", type=float, default=0.7,
			help="combined score threshold for HIGH drowsiness")

ap.add_argument("--show-metrics", action='store_true',
			help="show detailed eye and mouth metrics on screen")

args = vars(ap.parse_args())


# Normalize weights
w_eye = float(args["eye_weight"])
w_mouth = float(args["mouth_weight"])
wsum = w_eye + w_mouth
if wsum <= 0:
	w_eye, w_mouth, wsum = 0.7, 0.3, 1.0
w_eye /= wsum
w_mouth /= wsum


# Initialize MediaPipe Face Mesh
print("[INFO] Loading MediaPipe Face Mesh for eye+mouth combined score...")
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

# Mouth landmark indices
LIPS_OUTER = [
	61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
	291, 409, 270, 269, 267, 0, 37, 39, 40, 185
]
LIPS_INNER = [
	78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
	308, 415, 310, 311, 312, 13, 82, 81, 80, 191
]
MOUTH_IDXS = LIPS_OUTER + LIPS_INNER


# Video setup
if args["camera"]:
	# Treat input as camera index when --camera is set
	try:
		cam_index = int(args["input"])
	except ValueError:
		cam_index = 0
	vs = cv2.VideoCapture(cam_index)
else:
	vs = cv2.VideoCapture(args["input"])
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 960, 540)

writer = None
fps = vs.get(cv2.CAP_PROP_FPS)
if fps <= 0:
	fps = 25.0

# Eye and mouth histories
eye_window = max(1, int(fps * args["eye_window_sec"]))
mouth_window = max(1, int(fps * args["mouth_window_sec"]))

eye_history = deque(maxlen=eye_window)      # binary eye-drowsy per frame
mouth_history = deque(maxlen=mouth_window)  # binary mouth-open per frame

# Yawn tracking
yawn_timestamps = deque()  # Track yawn events in last 60 seconds
yawn_start_time = None
continuous_yawn_duration = 0.0
is_yawning = False

# Drowsiness level tracking
drowsiness_level = 0  # 0=ALERT, 1=PRE-ALERT, 2=MODERATE, 3=HIGH
critical_alert_timestamps = deque()  # Track critical alerts (Level 2+) in last 60 seconds
level_2_hold_until = None  # Timestamp to hold Level 2 for 20 seconds
last_critical_alert_frame = -100  # Track last frame when critical alert was added
previous_E_state = 0  # Track previous eye state to detect transitions
previous_yawn_trigger = False  # Track previous yawn trigger state

frame_count = 0

print("[INFO] Processing video with eye+mouth combined score...")
print(f"[INFO] Eye combined thresh: {args['eye_combined_thresh']:.2f}, window: {args['eye_window_sec']:.1f}s")
print(f"[INFO] Mouth combined thresh: {args['mouth_combined_thresh']:.2f}, window: {args['mouth_window_sec']:.1f}s")
print(f"[INFO] Weights -> eye: {w_eye:.2f}, mouth: {w_mouth:.2f}; medium: {args['medium_thresh']:.2f}, high: {args['high_thresh']:.2f}")
print(f"[INFO] Multi-level strategy: Yawn frequency (3/min), continuous yawn (3s), critical hold (3 alerts/min = 20s Level 2)")



while True:
	grabbed, frame = vs.read()
	if not grabbed:
		break

	frame_count += 1
	current_time = time.time()
	frame = imutils.resize(frame, width=960)
	frame_copy = frame.copy()
	gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
	h, w, _ = frame_copy.shape

	frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
	results = face_mesh.process(frame_rgb)

	if results.multi_face_landmarks:
		for face_landmarks in results.multi_face_landmarks:
			# All landmarks
			landmarks = []
			for landmark in face_landmarks.landmark:
				x = int(landmark.x * w)
				y = int(landmark.y * h)
				landmarks.append((x, y))

			# Eyes
			left_eye = [landmarks[i] for i in LEFT_EYE_IDXS]
			right_eye = [landmarks[i] for i in RIGHT_EYE_IDXS]
			left_ear = eye_aspect_ratio(left_eye)
			right_ear = eye_aspect_ratio(right_eye)
			avg_ear = (left_ear + right_ear) / 2.0

			left_perim = eye_perimeter_ratio(left_eye)
			right_perim = eye_perimeter_ratio(right_eye)
			avg_perim = (left_perim + right_perim) / 2.0

			left_area = eye_area_ratio(left_eye)
			right_area = eye_area_ratio(right_eye)
			avg_area = (left_area + right_area) / 2.0

			left_int = compute_eye_intensity(gray, left_eye, method=args["eye_intensity"])
			right_int = compute_eye_intensity(gray, right_eye, method=args["eye_intensity"])
			avg_int = (left_int + right_int) / 2.0

			eye_score = combined_eye_score(avg_ear, avg_perim, avg_area, avg_int)
			frame_eye_drowsy = 1 if eye_score > args["eye_combined_thresh"] else 0
			eye_history.append(frame_eye_drowsy)
			if len(eye_history) > 0:
				eye_drowsy_frac = sum(eye_history) / float(len(eye_history))
				E_state = 1 if eye_drowsy_frac >= args["eye_majority_frac"] else 0
			else:
				eye_drowsy_frac = 0.0
				E_state = 0

			# Mouth
			mouth = [landmarks[i] for i in MOUTH_IDXS]
			mar = mouth_aspect_ratio(mouth)
			opening_ratio = mouth_opening_ratio(mouth)
			perim_ratio = mouth_perimeter_ratio(mouth)
			area_ratio = mouth_area_ratio(mouth)
			mouth_score = combined_mouth_score(mar, opening_ratio, perim_ratio, area_ratio)
			frame_mouth_open = 1 if mouth_score > args["mouth_combined_thresh"] else 0
			mouth_history.append(frame_mouth_open)
			if len(mouth_history) > 0:
				mouth_open_frac = sum(mouth_history) / float(len(mouth_history))
				M_state = 1 if mouth_open_frac >= args["mouth_majority_frac"] else 0
			else:
				mouth_open_frac = 0.0
				M_state = 0

			# Track yawning events
			current_yawning = mouth_score > args["mouth_combined_thresh"]
			
			if current_yawning and not is_yawning:
				# Start of new yawn
				is_yawning = True
				yawn_start_time = current_time
			elif current_yawning and is_yawning:
				# Continuing yawn - track duration
				continuous_yawn_duration = current_time - yawn_start_time
			elif not current_yawning and is_yawning:
				# End of yawn - record if it was significant
				yawn_duration = current_time - yawn_start_time
				if yawn_duration > 0.5:  # At least 0.5 seconds to count as yawn
					yawn_timestamps.append(current_time)
				is_yawning = False
				yawn_start_time = None
				continuous_yawn_duration = 0.0
			
			# Clean up old yawn timestamps (keep last 60 seconds)
			while yawn_timestamps and (current_time - yawn_timestamps[0]) > 60.0:
				yawn_timestamps.popleft()
			
			# Count yawns in last minute
			yawn_count_per_minute = len(yawn_timestamps)
			
			# Calculate yawn frequency score
			yawn_frequency_trigger = yawn_count_per_minute >= 3
			continuous_yawn_trigger = continuous_yawn_duration >= 3.0
			current_yawn_trigger = yawn_frequency_trigger or continuous_yawn_trigger
			
			# Clean up old critical alerts (keep last 60 seconds)
			while critical_alert_timestamps and (current_time - critical_alert_timestamps[0]) > 60.0:
				critical_alert_timestamps.popleft()
			
			# Count critical alerts in last minute
			critical_alert_count = len(critical_alert_timestamps)
			
			# Determine drowsiness level with priority system
			# Priority 1: Eyes closed (highest priority - overrides hold timers)
			if E_state == 1:
				# Eyes are closed - immediate high priority
				if eye_drowsy_frac >= 0.8:
					new_level = 3  # HIGH - Eyes severely closed
				elif eye_drowsy_frac >= 0.5:
					new_level = 2  # MODERATE - Eyes significantly closed
				else:
					new_level = 1  # PRE-ALERT - Eyes starting to close
				
				# Add critical alert ONLY on transition from open to closed (Level 2+)
				# Detect transition: previous state was 0 (open) and current is 1 (closed)
				if new_level >= 2 and previous_E_state == 0:
					critical_alert_timestamps.append(current_time)
					last_critical_alert_frame = frame_count
				
				drowsiness_level = new_level
				# Clear hold timer when eyes are actively closed
				if new_level >= 2:
					level_2_hold_until = None
				
			# Priority 2: Yawning patterns (when eyes are open)
			elif current_yawn_trigger:
				# Yawn frequency: 3+ times in minute OR continuous yawn for 3+ seconds
				# This triggers MODERATE level for 20 seconds
				drowsiness_level = 2
				# Add critical alert ONLY on NEW yawn trigger (transition)
				if not previous_yawn_trigger:
					critical_alert_timestamps.append(current_time)
					last_critical_alert_frame = frame_count
					level_2_hold_until = current_time + 20.0
					
			elif level_2_hold_until and current_time < level_2_hold_until:
				# Hold Level 2 for 20 seconds after yawn trigger (only if eyes are open)
				drowsiness_level = 2
				
			# Priority 3: Check for multiple critical alerts
			elif critical_alert_count >= 3:
				# 3+ critical alerts in last minute - maintain Level 2 for 20 seconds
				drowsiness_level = 2
				if level_2_hold_until is None or current_time >= level_2_hold_until:
					level_2_hold_until = current_time + 20.0
			else:
				# No significant drowsiness detected
				drowsiness_level = 0
				level_2_hold_until = None
			
			# Update state tracking for next frame
			previous_E_state = E_state
			previous_yawn_trigger = current_yawn_trigger
			
			# Map drowsiness level to status
			if drowsiness_level == 3:
				status_text = "HIGH DROWSY"
				status_color = (0, 0, 255)  # Red
			elif drowsiness_level == 2:
				status_text = "MODERATE DROWSY"
				status_color = (0, 140, 255)  # Orange
			elif drowsiness_level == 1:
				status_text = "PRE-ALERT"
				status_color = (0, 255, 255)  # Yellow
			else:
				status_text = "ALERT"
				status_color = (0, 255, 0)  # Green

			# Draw contours
			left_eye_np = np.array(left_eye, dtype=np.int32)
			right_eye_np = np.array(right_eye, dtype=np.int32)
			cv2.polylines(frame_copy, [left_eye_np], True, (0, 255, 0), 2)
			cv2.polylines(frame_copy, [right_eye_np], True, (0, 255, 0), 2)

			mouth_outer = np.array(mouth[:20], dtype=np.int32)
			mouth_inner = np.array(mouth[20:], dtype=np.int32)
			cv2.polylines(frame_copy, [mouth_outer], True, (0, 255, 0), 2)
			cv2.polylines(frame_copy, [mouth_inner], True, (255, 0, 0), 2)

			# Text overlay - Status and Level
			y_pos = 25
			cv2.putText(frame_copy, f"Level {drowsiness_level}: {status_text}",
				(10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
			
			# Eye metrics
			y_pos += 30
			cv2.putText(frame_copy, f"Eye Score: {eye_score:.3f} | Drowsy Frac: {eye_drowsy_frac:.2f} (State={E_state})",
				(10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			
			# Mouth/Yawn metrics
			y_pos += 25
			cv2.putText(frame_copy, f"Mouth Score: {mouth_score:.3f} | Open Frac: {mouth_open_frac:.2f} (State={M_state})",
				(10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
			
			# Yawn tracking
			y_pos += 25
			yawn_status = "[YAWNING]" if is_yawning else ""
			cv2.putText(frame_copy, f"Yawns/min: {yawn_count_per_minute} | Duration: {continuous_yawn_duration:.1f}s {yawn_status}",
				(10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
			
			# Critical alerts
			y_pos += 25
			hold_status = ""
			if level_2_hold_until and current_time < level_2_hold_until:
				remaining = level_2_hold_until - current_time
				hold_status = f" [HOLD: {remaining:.1f}s]"
			cv2.putText(frame_copy, f"Critical Alerts/min: {critical_alert_count}{hold_status}",
				(10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

			if args["show_metrics"]:
				y_pos += 30
				cv2.putText(frame_copy,
					f"EAR:{avg_ear:.3f} Perim:{avg_perim:.2f} Area:{avg_area:.3f} Int:{avg_int:.3f}",
					(10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
				y_pos += 20
				cv2.putText(frame_copy,
					f"MAR:{mar:.3f} Open:{opening_ratio:.2f} MPerim:{perim_ratio:.2f} MArea:{area_ratio:.2f}",
					(10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

	else:
		cv2.putText(frame_copy, "No face detected",
			(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# Writer
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