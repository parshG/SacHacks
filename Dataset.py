import os
import cv2 as cv
import matplotlib.pyplot as plt
import mediapipe as mp
import pandas as pd
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = '/Users/dayallenragunathan/Downloads/asl_dataset'

data_list = []
labels = []

count = 0
for directory in os.listdir(data):
    folder_path = os.path.join(data, directory)
    if os.path.isdir(folder_path):
        for img_path in os.listdir(folder_path):
            img_file = os.path.join(folder_path, img_path)
            img = cv.imread(img_file)

            # Skip invalid images
            if img is None:
                print(f"Skipping invalid image: {img_file}")
                continue
            
            # Change img to RGB(mediapipe takes rgb imgs for landmarks)
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            res = hands.process(img_rgb)

            if res.multi_hand_landmarks:
                landmark_coords = []

                for hand_landmarks in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                    if hand_landmarks.landmark:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            z= hand_landmarks.landmark[i].z

                            landmark_coords.append(x)
                            landmark_coords.append(y)
                            landmark_coords.append(z)

                        data_list.append(landmark_coords)
                        labels.append(directory)




f = open('asl_dataset.pickle', 'wb')
pickle.dump((data_list, labels), f)
f.close()