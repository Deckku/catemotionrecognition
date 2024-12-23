# Importation des bibliothèques nécessaires
import os
import shutil
import random
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Définition des répertoires
source_dir = '/content/drive/MyDrive/NAYA_DATA_AUG1X'
train_dir = '/content/drive/MyDrive/catemotionrecognitionbyaudio/datasets/train'
val_dir = '/content/drive/MyDrive/catemotionrecognitionbyaudio/datasets/val'

# Liste des classes
def classes():
    return ['Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind',
            'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning']

# Fonction de répartition des données
def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in classes():
        class_train_dir = os.path.join(train_dir, class_name)
        class_val_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_train_dir, exist_ok=True)
        os.makedirs(class_val_dir, exist_ok=True)

        class_dir = os.path.join(source_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"[AVERTISSEMENT] Le dossier {class_name} n'existe pas dans {source_dir}.")
            continue

        audio_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        if not audio_files:
            print(f"[AVERTISSEMENT] Aucun fichier audio trouvé dans {class_dir}.")
            continue

        random.shuffle(audio_files)
        train_size = int(len(audio_files) * split_ratio)
        train_files = audio_files[:train_size]
        val_files = audio_files[train_size:]

        for file in train_files:
            shutil.move(os.path.join(class_dir, file), os.path.join(class_train_dir, file))

        for file in val_files:
            shutil.move(os.path.join(class_dir, file), os.path.join(class_val_dir, file))

        print(f"[INFO] {len(train_files)} fichiers déplacés vers {class_train_dir}")
        print(f"[INFO] {len(val_files)} fichiers déplacés vers {class_val_dir}")

# Appel de la fonction pour diviser les données
split_data(source_dir, train_dir, val_dir)
