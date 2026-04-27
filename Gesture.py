import cv2
import mediapipe as mp
import numpy as np

def process_and_visualize_video(video_path):
    # Initialize MediaPipe Hands and Drawing utilities
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    landmark_data = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        frame_hands = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Draw the skeleton onto the frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Normalizing the landmark coordinates
                raw_coords = []
                for lm in hand_landmarks.landmark:
                    raw_coords.append([lm.x, lm.y, lm.z])
                
                coords_array = np.array(raw_coords)
                wrist_coords = coords_array[0] 
                normalized_coords = coords_array - wrist_coords

                frame_hands.append(normalized_coords)

        # Append normalized frame data to master list
        landmark_data.append(frame_hands or None)

        cv2.imshow('MediaPipe Hand Landmarks', frame)

        # Press 'q' or 'Esc' to exit the preview early
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    return landmark_data

if __name__ == "__main__":
    VIDEO_FILE = "ASL_alphabet.mp4"
    extracted_data = process_and_visualize_video(VIDEO_FILE)
    print(f"Processed {len(extracted_data)} frames.")