import cv2
import mediapipe as mp


def get_face_landmarks(image,draw=False):
    # read the input image
    img_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_mask =mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,min_detection_confidence=0.5)
    image_rows, image_cols, _ = image.shape
    results = face_mask.process(img_input_rgb)

    image_landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = min(int(landmark.x * image_cols), image_cols - 1)
                y = min(int(landmark.y * image_rows), image_rows - 1)
                image_landmarks.append((x, y))
            if draw:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_tesselation_style())
    return image_landmarks
    