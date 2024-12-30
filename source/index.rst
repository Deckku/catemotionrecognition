catemotionrecognition documentation master file, created by
sphinx-quickstart on Thu Dec  5 23:02:11 2024.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

Welcome to cat emotion recognition Project
==========================================

Table des Matières
==================
1. Résumé
2. Introduction
3. Reconnaissance des émotions du chat par la voix
4. Reconnaissance des émotions du chat par l’image
5. Reconnaissance de l’état de santé du chat par l’image
6. Combinaison des modèles
7. Déploiement
8. Conclusion

---

1. Résumé
=========
Notre projet consiste à élaborer un modèle de machine learning capable de diagnostiquer l’état et l’humeur de l’animal domestique, le chat, en temps réel. On a conçu 3 modèles. Le 1er permet de générer une prédiction sur l’humeur à partir des images capturées, le 2e consiste à détecter aussi l’humeur, mais cette fois à partir des audios comme input. Le 3e permet de faire une prédiction sur l’état du chat à partir des images, en identifiant si son état est normal ou s’il est malade. Ensuite, on a combiné ces modèles pour générer une prédiction sur des vidéos réelles captées du chat. Enfin, on a fait le déploiement sur Streamlit et une application bureau.

---

2. Introduction
===============
Avec l’avancée rapide des technologies en intelligence artificielle et en machine learning, de nouvelles opportunités se présentent pour améliorer le bien-être des animaux domestiques. Notre projet s’inscrit dans cette démarche en visant à développer une solution innovante capable de diagnostiquer en temps réel l’état de santé et l’humeur d’un animal domestique, en particulier le chat. Ce travail repose sur l’analyse de données multimodales telles que les images, les audios, et les vidéos pour fournir des informations précises et utiles aux propriétaires d’animaux.
À travers ce projet, nous avons conçu et combiné plusieurs modèles de machine learning pour détecter non seulement les émotions du chat, mais aussi son état de santé, et avons déployé la solution sur des plateformes accessibles comme Streamlit et une application de bureau. L’objectif final est d’offrir un outil pratique et efficace permettant une interaction enrichie entre les propriétaires et leurs animaux tout en veillant à leur bien-être.

---

3. Reconnaissance des émotions du chat par la voix
==================================================

Guide de préparation des données pour la reconnaissance d'émotions par audio
----------------------------------------------------------------------------

Environnement : Google Colaboratory
Bibliothèques nécessaires : os, shutil, random

1. Collecte des données
-----------------------
La dataset utilisée comprend 5938 fichiers audio répartis en 10 classes :
['Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind', 'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning'].
Cette dataset n'est pas directement disponible en ligne et a été obtenue via Monsieur Yagya Raj Pandeya.
On a trouvé sur le plateforme kaggle une dataset de taille 100 audios de même distribution de classe que l'on a laissé pour la phase de test.

2. Prétraitement des données
----------------------------
Le prétraitement des données inclut la division en ensembles d'entraînement (80%) et de validation (20%).

.. code-block:: python

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

3. Préparation des données audio
---------------------------------
Nettoyage, transformation en spectrogrammes, normalisation et encodage des étiquettes.

.. code-block:: python

    def clean_audio(y, sr, low_freq=200, high_freq=8000):
        y_filtered = librosa.effects.preemphasis(y)
        stft = librosa.stft(y_filtered, n_fft=2048, hop_length=512)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        stft_filtered = stft[mask, :]
        return librosa.istft(stft_filtered, hop_length=512)

    def extract_spectrogram(y, sr, max_pad_len=128):
        try:
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr / 2)
            log_S = librosa.power_to_db(S, ref=np.max)
            pad_width = max_pad_len - log_S.shape[1]
            return np.pad(log_S, ((0, 0), (0, pad_width)), mode='constant') if pad_width > 0 else log_S[:, :max_pad_len]
        except Exception as e:
            print(f"[ERREUR] Extraction du spectrogramme échouée : {e}")
            return None

    def load_audio_files(base_dir, max_pad_len=128):
        audio_data, labels = [], []
        for label in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, label)
            if not os.path.isdir(folder_path):
                continue

            for file in os.listdir(folder_path):
                if file.endswith('.mp3'):
                    try:
                        y, sr = librosa.load(os.path.join(folder_path, file), sr=22050)
                        y_cleaned = clean_audio(y, sr)
                        spectrogram = extract_spectrogram(y_cleaned, sr, max_pad_len)
                        if spectrogram is not None:
                            audio_data.append(spectrogram)
                            labels.append(label)
                    except Exception as e:
                        print(f"[ERREUR] Erreur lors du traitement de {file}: {e}")

        return np.array(audio_data), np.array(labels)

Traitement des ensembles : entraînement, validation et test
-----------------------------------------------------------

.. code-block:: python

    def process_data_for_all_sets(train_dir, val_dir, test_dir, max_pad_len=128):
        X_train, y_train = load_audio_files(train_dir, max_pad_len)
        X_val, y_val = load_audio_files(val_dir, max_pad_len)
        X_test, y_test = load_audio_files(test_dir, max_pad_len)

        mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
        X_train = (X_train - mean) / (std + 1e-8)
        X_val = (X_val - mean) / (std + 1e-8) if X_val.size > 0 else X_val
        X_test = (X_test - mean) / (std + 1e-8) if X_test.size > 0 else X_test

        label_encoder = LabelEncoder().fit(np.concatenate([y_train, y_val, y_test]))
        y_train = label_encoder.transform(y_train)
        y_val = label_encoder.transform(y_val) if y_val.size > 0 else []
        y_test = label_encoder.transform(y_test) if y_test.size > 0 else []

        return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

Visualisation de spectrogrammes aléatoires
------------------------------------------

.. code-block:: python

    def display_random_spectrograms(data, labels, class_names, num_samples=5):
        indices = random.sample(range(len(data)), min(num_samples, len(data)))
        for idx in indices:
            plt.imshow(data[idx], aspect='auto', origin='lower', cmap='viridis')
            plt.title(f"Classe: {class_names[labels[idx]]}")
            plt.colorbar(format='%+2.0f dB')
            plt.show()

    # Exemple de traitement des données
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = process_data_for_all_sets(
        train_dir, val_dir, '/content/drive/MyDrive/catemotionrecognitionbyaudio/datasets/test'
    )
    display_random_spectrograms(X_train, y_train, label_encoder.classes_)

Image des spectrogrammes combinés
---------------------------------

Voici une représentation visuelle des spectrogrammes combinés pour chaque classe :

.. image:: 123.jpg
    :alt: Spectrogrammes combinés
    :width: 800px
    :align: center

### Construction des modèles
On a conçu un modèle CNN.

### Évaluation des performances
Mesures et résultats des tests effectués sur le modèle.

---

4. Reconnaissance des émotions du chat par l’image
==================================================

### Collecte des données

Description des étapes pour collecter les images nécessaires.

### Prétraitement des données

Techniques utilisées pour nettoyer et préparer les images pour l’entraînement.

### Construction des modèles

Architecture et algorithmes choisis pour le modèle de reconnaissance des émotions basé sur l’image.

### Évaluation des performances

Mesures et résultats des tests effectués sur le modèle.

---

5. Reconnaissance de l’état de santé du chat par l’image
==========================================================

### Collecte des données

Description des étapes pour collecter des données sur l’état de santé des chats.

### Prétraitement des données

Techniques utilisées pour préparer les données visuelles pour l’entraînement.

### Construction des modèles

Détails des modèles utilisés pour prédire l’état de santé.

### Évaluation des performances

Analyse des performances du modèle.

---

6.  Combinaison des modèles
===========================
Stratégies utilisées pour combiner les modèles d’analyse vocale et visuelle afin de fournir des prédictions plus complètes.

---

7. Déploiement
==============
Étapes pour déployer la solution sur Streamlit et l’application de bureau, avec les outils et frameworks utilisés.

---

8. Conclusion
=============
Résumé des résultats obtenus, des défis rencontrés et des perspectives d’avenir pour le projet.

---

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 2
   :caption: Table des Matières
   résumé
   introduction
   reconnaissance_voix
   reconnaissance_image
   reconnaissance_santé
   combinaison
   déploiement
   conclusion

