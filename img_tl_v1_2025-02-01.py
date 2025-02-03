import cv2
import json
import os
import numpy as np
from deepface import DeepFace
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

def detect_faces(image_path):
    """Erkennt Gesichter im Bild und gibt ihre Positionen zurück."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Bilddatei nicht gefunden: {image_path}")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError("Fehler beim Laden des Bildes. Überprüfe den Dateipfad und die Bilddatei.")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces, img

def estimate_ages(image_path):
    """Schätzt das Alter der erkannten Gesichter mit DeepFace."""
    try:
        analysis = DeepFace.analyze(image_path, actions=['age'], enforce_detection=False)
        return [entry['age'] for entry in analysis]
    except Exception as e:
        print(f"Fehler bei der Altersschätzung: {e}")
        return []

def find_reference_face(image_path):
    """Extrahiert das Gesicht der ersten Person aus dem ersten Bild als Referenz."""
    faces, img = detect_faces(image_path)
    if len(faces) == 0:
        raise ValueError("Kein Gesicht im Referenzbild gefunden.")
    
    x, y, w, h = faces[0]
    reference_face = img[y:y+h, x:x+w]
    return reference_face

def match_faces(reference_face, image_path):
    """Vergleicht Gesichter mit der Referenzperson und gibt deren Alter zurück."""
    try:
        result = DeepFace.find(img_path=image_path, db_path=image_path, enforce_detection=False)
        if result and len(result[0]) > 0:
            return estimate_ages(image_path)
    except Exception as e:
        print(f"Fehler bei der Gesichtserkennung: {e}")
    return []

def process_folder(folder_path):
    """Bearbeitet alle Bilddateien in einem Ordner und verfolgt eine Person."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Ordner nicht gefunden: {folder_path}")
    
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not image_files:
        raise FileNotFoundError("Keine Bilder im Ordner gefunden.")
    
    reference_face = find_reference_face(os.path.join(folder_path, image_files[0]))
    
    age_timeline = {}
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        output_file = os.path.splitext(image_path)[0]
        
        try:
            ages = match_faces(reference_face, image_path)
            if ages:
                age_timeline[image_file] = ages[0]
            
            with open(f"{output_file}.json", "w") as f:
                json.dump({"Alter": ages[0] if ages else None}, f, indent=4, default=lambda x: int(x) if isinstance(x, np.integer) else x)
            
            print(f"Analyse abgeschlossen für {image_file}. Ergebnisse gespeichert in {output_file}.json")
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {image_file}: {e}")
    
    with open(os.path.join(folder_path, "age_timeline.json"), "w") as f:
        json.dump(age_timeline, f, indent=4)

def main():
    folder_path = "/Users/blaubaer/Pictures/iCloud Fotos/"
    process_folder(folder_path)

if __name__ == "__main__":
    main()

