import cv2
import mediapipe as mp


def get_face_landmarks(image,draw=False):
    # read the input image
    img_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_mask =mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,min_detection_confidence=0.5)
    image_rows, image_cols, _ = image.shape
    results = face_mask.process(img_input_rgb)

    image_landmarks = []
    