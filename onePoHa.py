import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=10,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

csv_file = open('middle_knuckle_coordinates.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)

csv_writer.writerow(['frame', 'x', 'y', 'z'])

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark[9]
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)

            cv2.circle(frame, (x_px, y_px), 7, (0, 0, 255), -1)
            cv2.putText(frame, "Middle Knuckle", (x_px + 10, y_px - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            csv_writer.writerow([frame_count, lm.x, lm.y, lm.z])

    cv2.imshow('Middle Finger Knuckle Tracking', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
