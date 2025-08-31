import cv2
import numpy as np
import dlib
from imutils import face_utils
import winsound

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0,0,0)

def dist(a, b):
    return np.linalg.norm(a - b)

def eye_aspect_ratio(eye):
    A = dist(eye[1], eye[5])
    B = dist(eye[2], eye[4])
    C = dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

EAR_THRESH_SLEEP = 0.15
EAR_THRESH_DROWSY = 0.25
FRAME_THRESHOLD = 6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0,255,0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EAR_THRESH_SLEEP:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > FRAME_THRESHOLD:
                status = "SLEEPING !!!"
                color = (0,0,255)
                try:
                    winsound.Beep(2500, 800)
                except:
                    pass
        elif ear < EAR_THRESH_DROWSY:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > FRAME_THRESHOLD:
                status = "Drowsy !"
                color = (0,165,255)
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > FRAME_THRESHOLD:
                status = "Active :)"
                color = (0,255,0)

        for (x, y) in np.concatenate((left_eye, right_eye)):
            cv2.circle(face_frame, (x, y), 1, (255,255,255), -1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10,10), (360,120), (0,0,0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, f"Status: {status}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"EAR: {ear:.2f}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Sleep:{sleep}  Drowsy:{drowsy}  Active:{active}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
