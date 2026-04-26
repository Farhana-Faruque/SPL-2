import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from collections import deque
from typing import Deque, List, Dict, Any, Tuple, Optional
import time

import cv2
import mediapipe as mp

MIDDLE_KNUCKLE_IDX: int = 9

WINDOW_SIZE: int                = 15
STEP_SIZE: int                  = 5
MIN_REVERSALS: int              = 2
MIN_DISPLACEMENT: float         = 5.0
PCA_VARIANCE_THRESHOLD: float   = 0.90
RESIDUAL_RATIO_THRESHOLD: float = 0.10
IDLE_MOVEMENT_THRESHOLD: float  = 1.0
IDLE_TIME_THRESHOLD: float      = 1.0
SAMPLE_INTERVAL: float          = 0.4   # seconds between landmark samples
DAHUA_URL = "rtsp://admin:iit12345@10.160.40.2:554/cam/realmonitor?channel=3&subtype=0"


def initialize_mediapipe_hands(
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.7,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 0
) -> Tuple[mp.solutions.hands.Hands, Any]:
    """
    Initialise a MediaPipe Hands solution object.

    Args:
        max_num_hands (int): Maximum number of hands to detect (1 keeps it fast).
        min_detection_confidence (float): Minimum confidence for initial detection.
        min_tracking_confidence (float): Minimum confidence for landmark tracking.
        model_complexity (int): 0 = lite (faster), 1 = full (more accurate).

    Returns:
        Tuple[mp.solutions.hands.Hands, Any]:
            - hands: Configured Hands solution instance.
            - mp_drawing: MediaPipe drawing utilities (useful for debug overlays).
    """
    mp_hands = mp.solutions.hands
    
    hands = mp_hands.Hands(
        static_image_mode=False,          # Treat input as a continuous video stream
        max_num_hands=max_num_hands,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    mp_drawing = mp.solutions.drawing_utils
    return hands, mp_drawing


def get_middle_knuckle_coords(
    frame: np.ndarray,
    hands: mp.solutions.hands.Hands,
    landmark_idx: int = MIDDLE_KNUCKLE_IDX,
) -> Optional[Tuple[float, float]]:
    """
    Detect a single hand in *frame* and return the pixel coordinates of the
    requested landmark (default: middle-finger MCP = knuckle, index 9).

    The function converts the frame to RGB internally; the caller may keep
    working with the original BGR frame.

    Args:
        frame (np.ndarray): BGR image captured from OpenCV.
        hands (mp.solutions.hands.Hands): Initialised MediaPipe Hands instance.
        landmark_idx (int): Which hand landmark to extract (0-20).
                            9 → MIDDLE_FINGER_MCP (middle knuckle).

    Returns:
        Optional[Tuple[float, float]]:
            (x_px, y_px) in pixel coordinates if a hand is detected, else None.
    """
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False          # Minor performance hint
    results = hands.process(rgb)
    rgb.flags.writeable = True

    if not results.multi_hand_landmarks:
        return None  # No hand visible in this frame

    hand_landmarks = results.multi_hand_landmarks[0]
    lm = hand_landmarks.landmark[landmark_idx]

    x_px = lm.x * w
    y_px = lm.y * h
    return x_px, y_px


def run_cctv_stream(
    processor: "StreamProcessor",
    source: Any = 0,
    sample_interval: float = SAMPLE_INTERVAL,
    show_preview: bool = True,
    stop_key: str = "q",
) -> None:
    """
    Open a CCTV / webcam stream, detect the middle knuckle every
    *sample_interval* seconds, and feed the (timestamp, x, y) coordinate
    directly into *processor*.

    Args:
        processor (StreamProcessor): The StreamProcessor that classifies windows.
        source (Any): Camera index (int, e.g. 0) or an RTSP/HTTP URL string,
                      e.g. "rtsp://user:pass@192.168.1.10:554/stream".
        sample_interval (float): Seconds between successive landmark samples
                                 (default 0.4 s → 2.5 samples per second).
        show_preview (bool): If True, display a live OpenCV window with a
                             dot drawn on the detected knuckle and the latest
                             classification overlaid.  Set False for headless
                             (server / no-display) deployments.
        stop_key (str): Keyboard key (lower-case) that stops the loop when
                        the preview window is focused (default "q").

    Raises:
        RuntimeError: If the video source cannot be opened.

    Notes:
        - The function blocks until the user presses *stop_key* or the stream
          ends.  Run it in a dedicated thread / process for non-blocking use.
        - For IP cameras that require a different codec, pass
          `source="rtsp://..."` and ensure OpenCV is built with FFMPEG support.
    """
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video source: {source!r}. "
            "Check the camera index or RTSP URL."
        )
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(
        f"[CCTV] Stream opened: {frame_w}×{frame_h} @ "
        f"{native_fps:.1f} fps  |  sampling every {sample_interval}s"
    )

    
    hands, mp_drawing = initialize_mediapipe_hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        model_complexity=0,  
    )

    last_sample_time: float = 0.0   # wall-clock time of last successful sample
    missed_frames: int = 0          # consecutive frames without a hand

    KNUCKLE_COLOUR   = (0, 255, 127)   # BGR: spring green
    IDLE_COLOUR      = (60, 60, 220)   # BGR: muted red
    HARMONIC_COLOUR  = (50, 220, 50)   # BGR: green
    TEXT_FONT        = cv2.FONT_HERSHEY_SIMPLEX

    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CCTV] Stream ended or frame dropped.")
                break

            now = time.time()

    
            if now - last_sample_time >= sample_interval:
                coords = get_middle_knuckle_coords(frame, hands)

                if coords is not None:
                    x_px, y_px = coords
                    processor.add_data_point(now, x_px, y_px)
                    last_sample_time = now
                    missed_frames = 0

                    if show_preview:
                        
                        cv2.circle(frame, (int(x_px), int(y_px)), 8,
                                   KNUCKLE_COLOUR, -1)
                        cv2.circle(frame, (int(x_px), int(y_px)), 10,
                                   (255, 255, 255), 2)   # white ring

                else:
                    missed_frames += 1
                    if missed_frames >= 3:  # 3 × 0.4 s = 1.2 s with no hand
                        print(
                            f"[CCTV] No hand detected for "
                            f"{missed_frames * sample_interval:.1f}s"
                        )

           
            if show_preview:
                latest = processor.get_latest_result()
                if latest:
                    cat   = latest["category"]
                    src   = latest.get("source", "")
                    rev   = latest.get("reversals", 0)
                    label = (
                        f"HARMONIC  rev={rev}"
                        if cat == 1
                        else f"NON-HARMONIC  [{src}]"
                    )
                    colour = HARMONIC_COLOUR if cat == 1 else IDLE_COLOUR
                    cv2.rectangle(frame, (0, 0), (340, 36), (0, 0, 0), -1)
                    cv2.putText(frame, label, (8, 24),
                                TEXT_FONT, 0.65, colour, 2, cv2.LINE_AA)

           
                if now - last_sample_time < 0.05:
                    cv2.circle(frame, (frame_w - 16, 16), 7,
                               (255, 255, 0), -1)  # yellow = just sampled

                cv2.imshow("CCTV — Middle Knuckle Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord(stop_key):
                    print(f"[CCTV] '{stop_key}' pressed — stopping.")
                    break

    finally:
        
        cap.release()
        hands.close()
        if show_preview:
            cv2.destroyAllWindows()

        total = len(processor.get_results())
        harmonic = sum(
            1 for r in processor.get_results() if r["category"] == 1
        )
        print(
            f"[CCTV] Session ended — {total} windows classified, "
            f"{harmonic} harmonic, {total - harmonic} non-harmonic."
        )


def check_movement(
    projection: np.ndarray,
    times: np.ndarray,
    min_reversals: int,
    min_displacement: float
) -> Tuple[bool, int]:
    """
    Checks for significant movement and direction reversals within a projection.

    Args:
        projection (np.ndarray): 1D array of projected coordinates (e.g., from PCA).
        times (np.ndarray): 1D array of timestamps corresponding to the projection.
        min_reversals (int): Minimum number of reversals required to pass.
        min_displacement (float): Minimum velocity (px/sec) to consider as actual movement.

    Returns:
        Tuple[bool, int]: A tuple containing:
                          - bool: True if the movement heuristic passes, False otherwise.
                          - int: The total count of significant direction reversals.
    """
    if len(times) < 2:
        return False, 0

    dt = np.diff(times)
    dt[dt == 0] = 1e-6
    velocity = np.diff(projection) / dt

    significant_velocity = velocity[np.abs(velocity) >= min_displacement]

    if len(significant_velocity) < 2:
        return False, 0

    reversals = np.sum(np.diff(np.sign(significant_velocity)) != 0)
    return reversals >= min_reversals, int(reversals)


def check_pca_deviation(
    coords: np.ndarray,
    variance_threshold: float,
    residual_ratio_threshold: float
) -> Tuple[bool, float, float, np.ndarray]:
    """
    Checks for linearity and consistency of movement using PCA.

    Args:
        coords (np.ndarray): 2D array of (x, y) coordinates.
        variance_threshold (float): Minimum explained variance ratio for the first PCA component.
        residual_ratio_threshold (float): Maximum ratio of residual std dev to projection range.

    Returns:
        Tuple[bool, float, float, np.ndarray]: A tuple containing:
                                              - bool: True if the PCA heuristic passes.
                                              - float: Explained variance ratio of first component.
                                              - float: Residual ratio.
                                              - np.ndarray: Projection onto the first component.
    """
    if len(coords) < 2:
        return False, 0.0, float("inf"), np.array([])

    pca = PCA(n_components=2)
    pca.fit(coords)

    variance_ratio = pca.explained_variance_ratio_[0]
    projection     = pca.transform(coords)[:, 0]
    residuals      = pca.transform(coords)[:, 1]

    projection_range = np.ptp(projection)
    residual_ratio   = (
        np.std(residuals) / projection_range
        if projection_range > 0
        else float("inf")
    )

    # passed = (variance_ratio >= variance_threshold) and \
    #          (residual_ratio <= residual_ratio_threshold)
    passed = (variance_ratio >= variance_threshold) and (residual_ratio <= residual_ratio_threshold)

    return passed, variance_ratio, residual_ratio, projection


def classify_window(
    coords: np.ndarray,
    times: np.ndarray,
    pca_variance_threshold: float   = PCA_VARIANCE_THRESHOLD,
    residual_ratio_threshold: float = RESIDUAL_RATIO_THRESHOLD,
    min_reversals: int              = MIN_REVERSALS,
    min_displacement: float         = MIN_DISPLACEMENT,
) -> Tuple[int, bool, bool, int, float, float]:
    """
    Classifies a single window of movement data based on two heuristics.
    """
    if len(coords) < 3:
        return 2, False, False, 0, 0.0, float("inf")

    h2_passed, variance_ratio, residual_ratio, projection = check_pca_deviation(
        coords,
        variance_threshold=pca_variance_threshold,
        residual_ratio_threshold=residual_ratio_threshold,
    )

    h1_passed, reversal_count = check_movement(
        projection, times,
        min_reversals=min_reversals,
        min_displacement=min_displacement,
    )

    category = 1 if (h1_passed and h2_passed) else 2
    return category, h1_passed, h2_passed, reversal_count, variance_ratio, residual_ratio


class StreamProcessor:
    """
    Processes a real-time stream of (timestamp, x, y) data points.
    Maintains a sliding window, classifies movement, and detects idle periods.
    """
    def __init__(
        self,
        window_size: int              = WINDOW_SIZE,
        step_size: int                = STEP_SIZE,
        idle_movement_threshold: float = IDLE_MOVEMENT_THRESHOLD,
        idle_time_threshold: float    = IDLE_TIME_THRESHOLD,
    ):
        self.window_size            = window_size
        self.step_size              = step_size
        self.idle_movement_threshold = idle_movement_threshold
        self.idle_time_threshold    = idle_time_threshold

        # self.buffer: deque[Tuple[float, float, float]] = deque()
        from typing import Deque
        self.buffer: Deque[Tuple[float, float, float]] = deque()
        self.last_processed_idx  = -1
        self.results: List[Dict[str, Any]] = []
        self.last_activity_time: Optional[float] = None

    def add_data_point(self, timestamp: float, x: float, y: float) -> None:
        self.buffer.append((timestamp, x, y))

        while len(self.buffer) > self.window_size + self.last_processed_idx + self.step_size:
            self.buffer.popleft()
            self.last_processed_idx -= 1
            if self.last_processed_idx < -1:
                self.last_processed_idx = -1

        self._process_buffer()

    def _process_buffer(self) -> None:
        if len(self.buffer) < self.window_size:
            return

        current_buffer_end_idx = len(self.buffer) - 1
        if current_buffer_end_idx - self.last_processed_idx >= self.step_size:
            window_start_idx = current_buffer_end_idx - self.window_size + 1

            window_data = list(self.buffer)[window_start_idx : window_start_idx + self.window_size]
            times  = np.array([d[0] for d in window_data])
            coords = np.array([[d[1], d[2]] for d in window_data])

            category, h1, h2, reversals, var_ratio, res_ratio = classify_window(coords, times)

            if category == 1:
                self.last_activity_time = times[-1]

            self.results.append({
                "time_start":      times[0],
                "time_end":        times[-1],
                "category":        category,
                "h1_movement":     h1,
                "h2_pca":          h2,
                "reversals":       reversals,
                "variance_ratio":  round(var_ratio, 4),
                "residual_ratio":  round(res_ratio, 4),
                "source": "stream_working" if category == 1 else "stream_non_harmonic",
            })
            self.last_processed_idx = window_start_idx

        self._check_idle()

    def _check_idle(self) -> None:
        if len(self.buffer) < 2:
            return

        current_time = self.buffer[-1][0]
        if self.last_activity_time is None or \
                (current_time - self.last_activity_time > self.idle_time_threshold):

            recent_points  = list(self.buffer)[-min(self.window_size, len(self.buffer)):]
            if len(recent_points) < 2:
                return

            recent_coords  = np.array([[d[1], d[2]] for d in recent_points])
            displacement   = np.sqrt(np.sum(np.diff(recent_coords, axis=0) ** 2))

            if displacement < self.idle_movement_threshold:
                if not self.results or self.results[-1]["source"] != "idle":
                    self.results.append({
                        "time_start":     self.buffer[0][0],
                        "time_end":       current_time,
                        "category":       2,
                        "h1_movement":    False,
                        "h2_pca":         False,
                        "reversals":      0,
                        "variance_ratio": None,
                        "residual_ratio": None,
                        "source":         "idle",
                    })

    def get_results(self) -> List[Dict[str, Any]]:
        return self.results

    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        return self.results[-1] if self.results else None

      # Set False to run the synthetic simulation
def simulate_mediapipe_stream(processor: StreamProcessor, duration: int = 60, fps: int = 30) -> None:
    """
    Simulates a real-time MediaPipe data stream (kept for offline testing).
    """
    print("\n--- Starting MediaPipe Stream Simulation ---")
    start_time  = time.time()
    frame_count = 0

    for i in range(fps * 10):
        current_time = start_time + (i / fps)
        x = 50 + 20 * np.sin(current_time * 2)
        y = 50 + 10 * np.cos(current_time * 1.5)
        processor.add_data_point(current_time, x, y)
        frame_count += 1
        time.sleep(1 / fps)
        if frame_count % (fps * 2) == 0:
            r = processor.get_latest_result()
            if r:
                print(f"Time: {current_time:.2f}  Cat={r['category']}  src={r['source']}")

    print("\n--- Simulating Idle Period ---")
    idle_start = frame_count
    for i in range(fps * 5):
        current_time = start_time + (idle_start + i) / fps
        x = 70 + np.random.rand() * 0.5
        y = 60 + np.random.rand() * 0.5
        processor.add_data_point(current_time, x, y)
        frame_count += 1
        time.sleep(1 / fps)
        if frame_count % (fps * 2) == 0:
            r = processor.get_latest_result()
            if r:
                print(f"Time: {current_time:.2f}  Cat={r['category']}  src={r['source']}")

    print("\n--- Simulating More Movement ---")
    move_start = frame_count
    for i in range(fps * 10):
        current_time = start_time + (move_start + i) / fps
        x = 100 + 30 * np.cos(current_time * 3)
        y = 80  + 15 * np.sin(current_time * 2.5)
        processor.add_data_point(current_time, x, y)
        frame_count += 1
        time.sleep(1 / fps)
        if frame_count % (fps * 2) == 0:
            r = processor.get_latest_result()
            if r:
                print(f"Time: {current_time:.2f}  Cat={r['category']}  src={r['source']}")

    print("\n--- Simulation Ended ---")
    print("Total results:", len(processor.get_results()))



if __name__ == "__main__":
    stream_processor = StreamProcessor(
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        idle_movement_threshold=IDLE_MOVEMENT_THRESHOLD,
        idle_time_threshold=IDLE_TIME_THRESHOLD,
    )

    USE_LIVE_CAMERA = True    

    if USE_LIVE_CAMERA:
        
        run_cctv_stream(
            processor=stream_processor,
            source=DAHUA_URL,
            sample_interval=SAMPLE_INTERVAL,   # 0.4 s
            show_preview=True,
        )
    else:
        # ── Synthetic simulation mode (no camera required) ────────────────────
        simulate_mediapipe_stream(stream_processor, duration=25, fps=30)