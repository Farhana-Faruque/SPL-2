import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands = 10,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5   
)
mp_drawing = mp.solutions.drawing_utils

video_path = 'new.mp4'  
cap = cv2.VideoCapture(video_path)

csv_file = open('middle_knuckle_coordinates.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)

csv_writer.writerow(['frame', 'hand_id', 'x', 'y', 'z']) 

cv2.namedWindow('Middle Finger Knuckle Tracking', cv2.WINDOW_NORMAL)

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Reached end of video or failed to read frame.")
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        print(f"Frame {frame_count}: Detected {len(results.multi_hand_landmarks)} hands")
        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            lm = hand_landmarks.landmark[9]
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)

            cv2.circle(frame, (x_px, y_px), 7, (0, 0, 255), -1)
            cv2.putText(frame, f"Hand {hand_id}: Middle Knuckle", (x_px + 10, y_px - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            csv_writer.writerow([frame_count, hand_id, lm.x, lm.y, lm.z])
    else:
        print(f"Frame {frame_count}: No hands detected")

    cv2.imshow('Middle Finger Knuckle Tracking', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
csv_file.close()
cv2.destroyAllWindows()