import cv2
import mediapipe as mp
import pandas as pd

# Function to extract landmarks from a frame
def extract_landmarks(hand_landmarks):
    landmarks = []
    for point in hand_landmarks.landmark:
        landmarks.append((point.x, point.y, point.z))
    return landmarks

# Main function for capturing and saving hand gesture landmarks
def capture_landmarks(output_csv, num_samples_per_class=20):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    cap = cv2.VideoCapture(0)  # Change to 1 if you want to use an external camera

    columns = []
    for i in range(21):  # 21 landmarks per hand
        columns.append(f'x_{i}')
        columns.append(f'y_{i}')
        columns.append(f'z_{i}')

    columns.append('class')

    data = pd.DataFrame(columns=columns)

    while True:
        class_name = input("Enter the class name for this gesture (press 'q' to quit): ")
        if class_name.lower() == 'q':
            break
        
        sample_count = 0

        while sample_count < num_samples_per_class:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the image horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to get hand landmarks
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = extract_landmarks(hand_landmarks)

                    # Append landmarks and class to the dataframe
                    landmarks_flat = [coord for point in landmarks for coord in point]
                    landmarks_flat.append(class_name)
                    # data = data.append(pd.Series(landmarks_flat, index=data.columns), ignore_index=True)
                    data.loc[len(data)] = landmarks_flat
                    # Draw landmarks on the frame
                    for point in landmarks:
                        x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                    sample_count += 1

            # Display the frame
            cv2.imshow('Hand Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the VideoCapture and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Save the collected data to a CSV file
    data.to_csv(output_csv, index=False)
    print(f"Landmarks saved to {output_csv}")

if __name__ == "__main__":
    output_csv = "hand_gesture_landmarks1.csv"
    num_samples_per_class = 100
    capture_landmarks(output_csv, num_samples_per_class)
