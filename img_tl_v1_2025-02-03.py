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

def process_folder(folder_path):
    """Bearbeitet alle Bilddateien in einem Ordner und zeigt das Alter direkt auf den Bildern an."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Ordner nicht gefunden: {folder_path}")
    
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not image_files:
        raise FileNotFoundError("Keine Bilder im Ordner gefunden.")
    
    age_timeline = {}
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        output_file = os.path.splitext(image_path)[0]
        
        try:
            faces, img = detect_faces(image_path)
            ages = estimate_ages(image_path)
            
            if len(ages) != len(faces):
                print("Warnung: Anzahl erkannter Gesichter stimmt nicht mit Altersschätzungen überein.")
            
            for (x, y, w, h), age in zip(faces, ages):
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"{age} Jahre", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                age_timeline[image_file] = age
            
            annotated_image_path = f"{output_file}_annotated.jpg"
            cv2.imwrite(annotated_image_path, img)
            
            with open(f"{output_file}.json", "w") as f:
                json.dump({"Alter": ages}, f, indent=4, default=lambda x: int(x) if isinstance(x, np.integer) else x)
            
            print(f"Analyse abgeschlossen für {image_file}. Ergebnisse gespeichert in {output_file}.json und {annotated_image_path}")
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {image_file}: {e}")
    
    with open(os.path.join(folder_path, "age_timeline.json"), "w") as f:
        json.dump(age_timeline, f, indent=4)

def main():
    folder_path = "/Users/blaubaer/Pictures/iCloud Fotos/"
    process_folder(folder_path)

if __name__ == "__main__":
    main()

