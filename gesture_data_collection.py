import os
import cv2

# Define the directory to store the collected data
DATA_DIRECTORY = './data'
if not os.path.exists(DATA_DIRECTORY):
    os.makedirs(DATA_DIRECTORY)

# Define the number of classes and dataset size
NUM_CLASSES = 37
DATASET_SIZE = 500

# Initialize the camera capture
cap = cv2.VideoCapture(0)

# Iterate over each class for data collection
for class_index in range(NUM_CLASSES):
    # Create a directory for each class if it doesn't exist
    class_directory = os.path.join(DATA_DIRECTORY, str(class_index))
    if not os.path.exists(class_directory):
        os.makedirs(class_directory)

    print('Collecting data for class {}'.format(class_index))

    # Wait for the user to be ready to collect data
    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Collect data for the current class
    counter = 0
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_directory, '{}.jpg'.format(counter)), frame)

        counter += 1

# Release the camera capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
