from main import RealtimeFaceRecognition

# Nama orang sesuai urutan folder (alfabetis)
class_names = ['izza', 'nanta','dani']

recognizer = RealtimeFaceRecognition(
    model_path='face_recognition_model.h5',
    class_names=class_names
)

# Test dengan webcam
recognizer.recognize_from_webcam()