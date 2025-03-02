import cv2
import mediapipe as mp
import numpy as np
import pickle 

model = pickle.load(open('model.p','rb'))

labels = {
  0:'0',
  1:'1',
  2:'2',
  3:'3',
  4:'4',
  5:'5',
  6:'6',
  7:'7',
  8:'8',
  9:'9',
  10:'a',
  11:'b',
  12:'c',
  13:'d',
  14:'e',
  15:'f',
  16:'g',
  17:'h',
  18:'i',
  19:'j',
  20:'k',
  21:'l',
  22:'m',
  23:'n',
  24:'o',
  25:'p',
  26:'q',
  27:'r',
  28:'s',
  29:'t',
  30:'u',
  31:'v',
  32:'w',
  33:'x',
  34:'y',
  35:'z'}

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
      
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    predicted_character = "?"  
    predicted_label = -1
    
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
      data = []

      for hand_landmarks in results.multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
          x = hand_landmarks.landmark[i].x
          y = hand_landmarks.landmark[i].y
          data.append(x)
          data.append(y)

      if len(data) == 42:
        print("Feature data:", data[:10])
        prediction = model.predict([np.asarray(data)])
        predicted_label = int(prediction[0])
        predicted_character = labels[predicted_label]
        print(f"Predicted Character: {predicted_character}")
      else:
        print(f"Feature mismatch: Expected 63, got {len(data)}")

    cv2.putText(image, f'Prediction: {predicted_character}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(image, f'Label: {predicted_label}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()