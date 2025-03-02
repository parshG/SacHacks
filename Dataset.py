import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
import random
from pathlib import Path

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data_path = '/Users/dayallenragunathan/Downloads/asl_dataset'

backgrounds_dir = Path('./backgrounds')
backgrounds_dir.mkdir(exist_ok=True)

background_images = []
if backgrounds_dir.exists():
    for bg_file in backgrounds_dir.glob('*.jpg'):
        bg = cv2.imread(str(bg_file))
        if bg is not None:
            background_images.append(bg)
    for bg_file in backgrounds_dir.glob('*.png'):
        bg = cv2.imread(str(bg_file))
        if bg is not None:
            background_images.append(bg)

debug_dir = Path('./debug_augmented')
debug_dir.mkdir(exist_ok=True)
save_debug_images = True

def segment_hand(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    hand = cv2.bitwise_and(image, image, mask=mask)
    
    return hand, mask

def generate_synthetic_background(shape):
    h, w = shape[:2]
    
    bg_type = random.randint(0, 4)
    
    if bg_type == 0:
        color = [random.randint(0, 255) for _ in range(3)]
        bg = np.ones((h, w, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)
    
    elif bg_type == 1:
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(w):
            color = [
                int(i * 255 / w), 
                int((w - i) * 255 / w), 
                random.randint(0, 255)
            ]
            bg[:, i] = color
    
    elif bg_type == 2:
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            color = [
                int(i * 255 / h), 
                random.randint(0, 255), 
                int((h - i) * 255 / h)
            ]
            bg[i, :] = color
    
    elif bg_type == 3: 
        bg = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    else:  
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        square_size = random.randint(10, 30)
        for i in range(0, h, square_size):
            for j in range(0, w, square_size):
                if (i//square_size + j//square_size) % 2 == 0:
                    color = [random.randint(100, 255) for _ in range(3)]
                else:
                    color = [random.randint(0, 100) for _ in range(3)]
                bg[i:i+square_size, j:j+square_size] = color
    
    return bg

def replace_background(image, mask, background):
    bg_resized = cv2.resize(background, (image.shape[1], image.shape[0]))
    
    inv_mask = cv2.bitwise_not(mask)
    
    fg = cv2.bitwise_and(image, image, mask=mask)
    
    bg = cv2.bitwise_and(bg_resized, bg_resized, mask=inv_mask)
    
    result = cv2.add(fg, bg)
    
    return result

def apply_augmentations(image):
    brightness = random.uniform(0.7, 1.3)
    contrast = random.uniform(0.7, 1.3)
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness*50)
    
    if random.random() < 0.3:
        blur_size = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    
    if random.random() < 0.3:
        noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
    
    return image

data_list = []
labels = []

augmentation_count = 0

for directory in os.listdir(data_path):
    folder_path = os.path.join(data_path, directory)
    if os.path.isdir(folder_path):
        print(f"Processing {directory}...")
        
        for img_path in os.listdir(folder_path):
            img_file = os.path.join(folder_path, img_path)
            img = cv2.imread(img_file)
            
            if img is None:
                print(f"Skipping invalid image: {img_file}")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res_original = hands.process(img_rgb)
            
            if res_original.multi_hand_landmarks:
                landmark_coords_original = []
                for hand_landmarks in res_original.multi_hand_landmarks:
                    if hand_landmarks.landmark:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            landmark_coords_original.append(x)
                            landmark_coords_original.append(y)
                
                if len(landmark_coords_original) == 42:  
                    data_list.append(landmark_coords_original)
                    labels.append(directory)
                
                hand_region, hand_mask = segment_hand(img)
                
                for aug_idx in range(5): 
                    if background_images and random.random() < 0.7:
                        bg = random.choice(background_images)
                    else:
                        bg = generate_synthetic_background(img.shape)
                    
                    augmented_img = replace_background(img, hand_mask, bg)
                    
                    augmented_img = apply_augmentations(augmented_img)
                    
                    augmented_rgb = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB)
                    res_augmented = hands.process(augmented_rgb)
                    
                    if res_augmented.multi_hand_landmarks:
                        landmark_coords_augmented = []
                        for hand_landmarks in res_augmented.multi_hand_landmarks:
                            if hand_landmarks.landmark:
                                for i in range(len(hand_landmarks.landmark)):
                                    x = hand_landmarks.landmark[i].x
                                    y = hand_landmarks.landmark[i].y
                                    landmark_coords_augmented.append(x)
                                    landmark_coords_augmented.append(y)
                        
                        if len(landmark_coords_augmented) == 42:
                            data_list.append(landmark_coords_augmented)
                            labels.append(directory)
                            augmentation_count += 1
                        
                        if save_debug_images and aug_idx == 0:
                            debug_path = debug_dir / f"{directory}_{os.path.basename(img_file)}_aug{aug_idx}.jpg"
                            
                            debug_img = augmented_img.copy()
                            for hand_landmark in res_augmented.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                    debug_img,
                                    hand_landmark,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style()
                                )
                            
                            cv2.imwrite(str(debug_path), debug_img)



with open('asl_dataset_enhanced.pickle', 'wb') as f:
    pickle.dump((data_list, labels), f)
