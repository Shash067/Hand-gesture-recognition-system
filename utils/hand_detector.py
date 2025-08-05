import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandDetector:
    def __init__(self, max_hands=1, detection_conf=0.7):
        self.hands = mp_hands.Hands(static_image_mode=False,
                                     max_num_hands=max_hands,
                                     min_detection_confidence=detection_conf)
        self.drawer = mp_drawing

    def process(self, image_rgb):
        return self.hands.process(image_rgb)

    def draw(self, frame, landmarks):
        self.drawer.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
