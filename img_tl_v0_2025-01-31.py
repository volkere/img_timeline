import os
import cv2
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from PIL import Image
from PIL.ExifTags import TAGS
import matplotlib.pyplot as plt

# ImageNet-Modell laden (ResNet50)
model = ResNet50(weights="imagenet")

def get_exif_date(image_path):
    """Versucht, das Erstellungsdatum aus den EXIF-Daten zu extrahieren."""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == "DateTimeOriginal":  # Aufnahmedatum suchen
                    return value
    except Exception as e:
        print(f"EXIF-Fehler für {image_path}: {e}")
    return None

def classify_image(image_path):
    """Verwendet ImageNet (ResNet50), um das Bild in eine Kategorie einzuordnen."""
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    labels = decode_predictions(preds, top=3)[0]  # Nimmt die Top-3 Kategorien
    return labels

def process_images(folder):
    """Liest Bilder aus einem Ordner, prüft Metadaten und klassifiziert sie mit ImageNet."""
    folder = os.path.abspath(folder)
    if not os.path.exists(folder):
        print(f"Ordner '{folder}' existiert nicht. Erstelle ihn...")
        os.makedirs(folder)
    
    timeline_data = []

    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder, file)
            print(f"Verarbeite: {image_path}")

            # 1️EXIF-Daten auslesen
            date_taken = get_exif_date(image_path)

            # 2️Falls kein Datum vorhanden → Bild mit ImageNet klassifizieren
            if not date_taken:
                predictions = classify_image(image_path)
                date_taken = f"Geschätzt anhand von ImageNet: {predictions[0][1]} ({predictions[0][2]*100:.2f}%)"

            # 3️Speichern der Daten
            timeline_data.append({"Bild": file, "Datum": date_taken})

    return timeline_data

def save_timeline(timeline_data, output_file):
    """Speichert die Timeline als CSV und JSON."""
    df = pd.DataFrame(timeline_data)
    df.to_csv(output_file + ".csv", index=False)
    with open(output_file + ".json", "w") as f:
        json.dump(timeline_data, f, indent=4)

def plot_timeline(timeline_data):
    """Erstellt eine grafische Zeitleiste."""
    dates = [entry["Datum"] for entry in timeline_data]
    images = [entry["Bild"] for entry in timeline_data]

    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(dates)), dates, marker="o", color="b")
    plt.xlabel("Bilder")
    plt.ylabel("Datum")
    plt.title("Bild-Timeline")
    plt.xticks(rotation=45)
    plt.show()

# 📂 Führe das Skript aus
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, "bilder")
output_file = "timeline"

timeline = process_images(folder_path)
save_timeline(timeline, output_file)
plot_timeline(timeline)

