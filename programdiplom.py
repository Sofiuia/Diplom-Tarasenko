
import cv2
import face_recognition
from utils.image_processing import preprocess_frame
from utils.dataset import get_known_faces
from config import Config

last_faces = []
frame_counter = 0

def recognize_face(encoding, known_enc, known_names):
    if not known_enc:
        return "Unknown"
    distances = face_recognition.face_distance(known_enc, encoding)
    best_index = distances.argmin()
    if distances[best_index] < Config.FACE_TOLERANCE:
        return known_names[best_index]
    return "Unknown"


def process_frame_ai(frame, small_frame):
    """AI режим """
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    known_enc, known_nam = get_known_faces()

    for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
        name = recognize_face(enc, known_enc, known_nam)

        top = int(top * 5)
        right = int(right * 5)
        bottom = int(bottom * 5)
        left = int(left * 5)
        
        color = (0, 255, 100) if name != "Unknown" else (0, 165, 255)
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 4)
        cv2.putText(frame, name, (left, top - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        cv2.putText(frame, "AI", (left, top - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
    return frame


def process_frame_haar(frame):
    """HAAR режим — оптимізований"""
    global frame_counter, last_faces
    frame_counter += 1

    if frame_counter % 3 != 0 and last_faces:
        for (x, y, w, h, name) in last_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 165, 0), 3)
            cv2.putText(frame, name, (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
            cv2.putText(frame, "HAAR", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)
        return frame

    gray = preprocess_frame(frame)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80))

    known_enc, known_nam = get_known_faces()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_faces = []

    for (x, y, w, h) in faces:
        name = "Unknown"
        if w > 100:
            face_img = rgb_frame[y:y+h, x:x+w]
            encodings = face_recognition.face_encodings(face_img)
            if encodings:
                name = recognize_face(encodings[0], known_enc, known_nam)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 165, 0), 3)
        cv2.putText(frame, name, (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        cv2.putText(frame, "HAAR", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)

        current_faces.append((x, y, w, h, name))

    last_faces = current_faces
    return frame


def process_frame_reserve(frame):
    gray = preprocess_frame(frame)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(frame, "RESERVE MODE", (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame