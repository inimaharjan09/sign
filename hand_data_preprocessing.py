import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the directory containing the image data
DATA_DIRECTORY = './data'

# Initialize lists to store extracted hand data and corresponding labels
hand_data = []
labels = []

# Iterate over each directory (class) in the data directory
for class_folder in os.listdir(DATA_DIRECTORY):
    # Iterate over each image file in the class directory
    for img_filename in os.listdir(os.path.join(DATA_DIRECTORY, class_folder)):
        # Initialize list to store hand landmarks data for each image
        hand_data_aux = []

        # Initialize lists to store x and y coordinates of hand landmarks
        x_coords = []
        y_coords = []

        # Read and convert image to RGB format
        img = cv2.imread(os.path.join(DATA_DIRECTORY, class_folder, img_filename))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)

        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            # Iterate over each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract x and y coordinates of each hand landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    # Append coordinates to respective lists
                    x_coords.append(x)
                    y_coords.append(y)

                # Normalize coordinates by subtracting minimum values
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    hand_data_aux.append(x - min(x_coords))
                    hand_data_aux.append(y - min(y_coords))

            # Append extracted hand data and corresponding label to lists
            hand_data.append(hand_data_aux)
            labels.append(class_folder)

# Save extracted data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'hand_data': hand_data, 'labels': labels}, f)