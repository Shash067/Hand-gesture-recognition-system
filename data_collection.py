import cv2
from utils.hand_detector import HandDetector
from utils.data_utils import save_landmarks

FILENAME = "dataset.csv"

detector = HandDetector()
cap = cv2.VideoCapture(0)

label = input("Enter label for this gesture: ")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            detector.draw(frame, hand_landmarks)
            cv2.putText(frame, "Press 's' to save", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            landmarks = hand_landmarks.landmark

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(10)
    if key == ord('s') and results.multi_hand_landmarks:
        save_landmarks(FILENAME, landmarks, label)
        print(f"Saved for label: {label}")
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
