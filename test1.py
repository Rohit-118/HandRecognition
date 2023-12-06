import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load the trained model and label encoder
# model = joblib.load("hgmodel_RandomForest.joblib")
# label_encoder = joblib.load("label_encoder1.joblib")

# model=joblib.load("hgmodel_KNN.joblib")

# model=joblib.load("hgmodel_SVM.joblib")

model=joblib.load("hgmodel_MLP.joblib")

label_encoder=joblib.load("StandardScaler.joblib")

# Function to extract landmarks from a frame
def extract_landmarks(hand_landmarks):
    landmarks = []
    for point in hand_landmarks.landmark:
        landmarks.extend([point.x, point.y, point.z])
    return np.array(landmarks)

# Main function for live prediction
def predict_live():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    cap = cv2.VideoCapture(0)  # Change to 1 if you want to use an external camera

    while cap.isOpened():
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

                # Make a prediction using the trained model
                prediction = model.predict([landmarks])[0]
                class_name = label_encoder.inverse_transform([prediction])[0]

                # draw landmarks on the frame
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            

                # Display the predicted class on the frame
                h, w, _ = frame.shape
                x, y = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
                cv2.putText(frame, f"Prediction: {class_name}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Hand Gesture Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_live()
