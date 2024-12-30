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

1. Collecte des données
-----------------------
Les images nécessaires ont été collectées à partir de diverses sources publiques et bases de données spécialisées, garantissant une diversité de visages et d'expressions émotionnelles. Des critères d’inclusion spécifiques, tels que la résolution et la qualité des images, ont été définis pour assurer la pertinence des données. Des autorisations ont été respectées pour les sources publiques afin de garantir un usage éthique.

2. Préparation des données image
---------------------------------
Nettoyage, transformation en spectrogrammes, normalisation et encodage des étiquettes.

.. code-block:: python

    def load_and_preprocess_images(image_dir):
    images = []
    labels = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(image_dir, class_name)
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(class_idx)
    return np.array(images), to_categorical(np.array(labels), num_classes=NUM_CLASSES)



3. Construction du modele
---------------------------------
Une architecture CNN (Convolutional Neural Network) a été choisie pour ses performances éprouvées dans le traitement d'images. Le modèle a été construit avec plusieurs couches convolutives suivies de couches de pooling et d’une couche dense finale. L’optimisation a été réalisée à l’aide de l’algorithme Adam, et des fonctions d’activation ReLU ont été utilisées.

.. code-block:: python

    def create_cnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

4. Classes de Reconnaissance des Émotions Basée sur l'Image
--------------------------------------------------------

Dans le cadre de l'analyse des émotions des chats à partir d'images, nous avons utilisé un modèle de réseau de neurones convolutifs (CNN) pour identifier les émotions des chats à partir de photos. Les classes d'émotions que le modèle est capable de reconnaître sont les suivantes :

1. **Angry (En colère)** : Cette émotion indique que le chat se sent en colère, souvent en réponse à une menace ou une perturbation. Les signes de colère peuvent inclure des oreilles pointées en arrière, des yeux dilatés et une posture tendue.

2. **Beg (Mendiant)** : Lorsque le chat cherche de l'attention ou de la nourriture, il peut adopter une posture de "mendicité". Cela inclut souvent une attitude de sollicitation, comme se frotter contre les jambes ou regarder fixement en direction de la nourriture.

3. **Annoyed (Agacé)** : Un chat peut être agacé lorsqu'il est dérangé, même de manière subtile. Les signes incluent souvent une posture tendue ou un regard distant.

4. **Frightened (Effrayé)** : Lorsque le chat ressent de la peur, il peut se figer, ses yeux deviennent plus larges et ses oreilles se replient. Il essaie souvent d'éviter la situation qui provoque cette peur.

5. **Happy (Heureux)** : Un chat heureux est souvent détendu, avec des yeux mi-clos et une posture douce. Le ronronnement est aussi un indicateur clé d'une émotion positive chez les chats.

6. **Normal (Normal)** : Un chat dans un état émotionnel "normal" montre des signes de calme et de confort. Il peut être dans une posture détendue sans aucune indication de stress ou d'agression.

7. **Sad (Triste)** : Un chat triste peut montrer des signes de dépression ou de désintérêt, comme une posture affaissée, des yeux mi-clos ou une perte d'appétit.

8. **Scared (Effrayé)** : Similaire à "Frightened", mais souvent avec un sentiment plus intense. Un chat effrayé peut être plus réactif et chercher à fuir.

9. **Under the Weather (Pas bien)** : Un chat "pas bien" peut sembler léthargique, avoir une posture plus repliée, et montrer moins d'intérêt pour son environnement, ce qui peut indiquer qu'il est malade ou fatigué.

10. **Curious (Curieux)** : Un chat curieux montre souvent des signes d'exploration, comme une attention accrue aux nouveaux objets ou environnements, avec des oreilles en avant et une posture droite.

11. **Playful (Joueur)** : Un chat joueur adopte une posture excitée, souvent avec les pattes avant étendues ou en train de sauter autour d'un objet, montrant une énergie ludique et curieuse.

Ces classes sont utilisées pour déterminer l'état émotionnel général du chat à partir de ses expressions faciales et de son comportement visible sur les images. L'objectif est de mieux comprendre le bien-être des chats et d'offrir une méthode non intrusive pour observer leurs émotions.


5. Reconnaissance de l’état de santé du chat par l’image
==========================================================

1. Collecte des données
-----------------------

Chaque image est lue, redimensionnée, normalisée et étiquetée avec une catégorie spécifique correspondant à l'état de santé.

2. Construction du modele
---------------------------------
Une architecture CNN (Convolutional Neural Network) a été choisie pour ses performances éprouvées dans le traitement d'images. Le modèle a été construit avec plusieurs couches convolutives suivies de couches de pooling et d’une couche dense finale. L’optimisation a été réalisée à l’aide de l’algorithme Adam, et des fonctions d’activation ReLU ont été utilisées.

.. code-block:: python

    def create_cnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



Dans ce cas, les images doivent être étiquetées pour deux classes principales :

**Sick** : Images représentant un chat malade, avec des signes visibles de maladie.
**Healthy** : Images représentant un chat en bonne santé, sans signes de maladie.

Les images peuvent provenir de diverses sources, mais elles doivent être étiquetées avec précision pour garantir la performance du modèle. Le prétraitement des données reste similaire à ce qui a été décrit précédemment, en redimensionnant et en normalisant les images.




---


6. Combinaison des modèles 
==========================

Le but de cette étape est de combiner les trois modèles de traitement d'image pour offrir une solution complète de classification vidéo. Ces modèles incluent :

**Modèle de reconnaissance des émotions du chat** : Ce modèle identifie les émotions du chat à partir d'images fixes, telles que l'angoisse, la joie, la peur, etc.
**Modèle de reconnaissance de l'état de santé du chat** : Ce modèle est chargé de déterminer si le chat est malade ou en bonne santé en analysant des images.
**Modèle de traitement vidéo** : Cette partie combine les prédictions des deux premiers modèles sur chaque image d'une vidéo pour fournir des résultats dynamiques (sur les émotions et l'état de santé) tout au long de la séquence vidéo.

Stratégies pour combiner les modèles:

Le défi ici est d'intégrer ces deux modèles (émotions et santé) dans une chaîne de traitement vidéo. Voici les principales étapes de la combinaison des modèles :

Extraction des images vidéo : Pour analyser une vidéo, il faut d'abord en extraire les images (frames). Ces images sont ensuite envoyées aux deux modèles pour obtenir des prédictions individuelles.

Traitement par le modèle d'émotions : Chaque image extraite de la vidéo est envoyée au modèle de reconnaissance des émotions. Le modèle génère une prédiction sur l'émotion du chat à partir de l'image.

Traitement par le modèle de santé : La même image est ensuite envoyée au modèle de reconnaissance de l'état de santé, qui prédit si le chat est malade ou non.

Fusion des résultats : Les prédictions des deux modèles peuvent être combinées pour donner un aperçu global de l'état de santé et des émotions du chat pendant la vidéo. Cela pourrait se faire par :

Moyenne ou pondération des résultats des deux modèles pour une prise de décision finale par frame.

Affichage des résultats combinés à chaque frame sous forme d'annotations (par exemple, afficher à la fois l'émotion du chat et son état de santé).

Suivi dynamique dans la vidéo : En utilisant une fenêtre temporelle (par exemple, sur plusieurs frames), vous pouvez suivre l'évolution des émotions et de l'état de santé du chat au cours du temps. Cette approche peut être utilisée pour détecter des changements dans les émotions ou l'état de santé du chat dans la vidéo.

.. code-block:: python

    def process_video(video_path):
        try:
            # Process video frame
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video file")
                return None, None, None, None

            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Could not read video frame")
                return None, None, None, None

            temp_frame = os.path.join(os.path.dirname(video_path), "temp_frame.jpg")
            cv2.imwrite(temp_frame, frame)
            
            # Process the frame for visual analysis
            emotion, sickness, confidence = process_image(temp_frame)
            if os.path.exists(temp_frame):
                os.remove(temp_frame)
            cap.release()

            # Extract and process audio
            audio_mood = None
            temp_audio = extract_audio(video_path)
            if temp_audio:
                audio_mood = process_audio(temp_audio)
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)
            else:
                messagebox.showwarning("Warning", "Could not analyze audio. Make sure the video contains audio.")

            if emotion and sickness:
                return emotion, sickness, confidence, audio_mood
            return None, None, None, None

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process video: {str(e)}")
            return None, None, None, None

7. Déploiement
==============
1. Vue d'ensemble
-----------------

L'application **Cat Mood & Health Analyzer** a été créée en utilisant **Tkinter**, une bibliothèque Python pour la création d'interfaces graphiques. L'interface est conçue pour être simple et intuitive, permettant à l'utilisateur de télécharger des fichiers multimédia (images, vidéos, ou audio) via un bouton de sélection. Lorsqu'un fichier est téléchargé, l'application traite le contenu à l'aide de modèles pré-entrainés pour analyser les émotions, l'état de santé et l'humeur du chat. Les résultats sont affichés dans l'interface, accompagnés d'un message personnalisé. L'application comprend également un système de gestion des erreurs pour informer l'utilisateur de tout problème de traitement. Enfin, l'application peut être convertie en un exécutable autonome à l'aide de **PyInstaller** pour une utilisation sans installation préalable de Python.

2. Aperçu
---------

Le Cat Mood & Health Analyzer est une application conviviale conçue pour analyser l'humeur et l'état de santé des chats à partir de divers fichiers multimédias, tels que des images, des vidéos et des enregistrements audio. En utilisant des modèles d'apprentissage automatique avancés, l'outil offre des informations sur les émotions et le bien-être de votre chat. L'application utilise trois modèles principaux :

1. **Modèle de reconnaissance des émotions** : Ce modèle prédit l'état émotionnel d'un chat à partir des images, en le classant dans des catégories telles que heureux, triste, en colère ou joueur.
2. **Modèle de détection de maladie** : En utilisant des images, ce modèle évalue si le chat est en bonne santé ou potentiellement malade, en fournissant une classification "normal" ou "malade".
3. **Modèle d'analyse de l'humeur audio** : Ce modèle traite les fichiers audio, tels que les miaulements ou les ronronnements, pour détecter différents états ou humeurs, tels que la colère, la joie et la douleur.

En téléchargeant une photo, une vidéo ou un enregistrement audio d'un chat, les utilisateurs peuvent recevoir une analyse détaillée comprenant l'état émotionnel du chat, son état de santé et son humeur audio, aidant ainsi les propriétaires à mieux comprendre les besoins de leur chat. L'interface graphique intuitive facilite l'interaction, et elle fournit des résultats en temps réel avec des retours détaillés basés sur l'analyse.

3. Fonctionnalités
------------------

- **Analyse d'image** : Téléchargez une image nette de votre chat pour la prédiction de l'émotion et de la santé.
- **Analyse vidéo** : Téléchargez des vidéos, et l'application traitera à la fois le contenu visuel et audio pour fournir une analyse plus complète.
- **Analyse audio** : Téléchargez des fichiers audio des sons de votre chat pour la détection de l'humeur.
- **Interface conviviale** : Le design simple et moderne le rend accessible à tous les amoureux des chats, des propriétaires occasionnels aux professionnels.
- **Retour en temps réel** : Recevez immédiatement un retour sur l'état émotionnel et de santé de votre chat avec des messages faciles à comprendre.

4. Commencer
------------

1. Lancez l'application.
2. Téléchargez une image, une vidéo ou un fichier audio de votre chat.
3. L'application analysera le fichier et fournira un rapport détaillé sur l'humeur, l'état de santé et les sons associés de votre chat.
4. Si une vidéo est téléchargée, les analyses visuelles et audio seront traitées.
5. Consultez les résultats et utilisez les informations pour mieux prendre soin de votre chat.


.. code-block:: python

    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import numpy as np
    import cv2
    import librosa
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    from PIL import Image, ImageTk
    import os
    import subprocess

    # Constants
    IMG_HEIGHT, IMG_WIDTH = 128, 128
    IMAGE_CLASSES = ["angry", "beg", "annoyed", "frightened", "happy", "normal", "sad", "scared", "under the weather", "curious", "playful"]
    SICKNESS_CLASSES = ["normal", "sick"]
    AUDIO_CLASSES = ['Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind', 'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning']

    # Load models
    try:
        image_model = load_model('catemotionrecognition\\progress\\maybe3.h5')
        sickness_model = load_model('catemotionrecognition\\progress\\sicknormalf.h5')
        audio_model = load_model('catemotionrecognition\\progress\\issa.h5')
    except:
        messagebox.showerror("Error", "Could not load models. Please check if model files exist.")
        exit()

    def get_friendly_message(emotion, sickness, confidence, mood=None):
        if confidence < 0.1:
            return "I'm not confident this is a cat image, or the image might be unclear. Please try uploading a clearer picture of a cat! 🐱"
        
        messages = {
            "happy": "Your cat seems to be in a great mood! ", "sad": "Aww, your cat might need some extra love and attention right now 💕",
            "angry": "Looks like someone woke up on the wrong side of the bed! 😾", "sick": "Your cat might not be feeling well. Consider a vet visit! 🏥",
            "normal": "Your cat appears to be healthy! 🌟", "beg": "Your cat is begging for something. Maybe it's time for a treat! 🍖",
            "annoyed": "Your cat seems annoyed. It might need some space. 😒", "frightened": "Your cat seems frightened. Try to comfort it. 😨",
            "scared": "Your cat is scared. It might need some reassurance. 😱", "under the weather": "Your cat seems under the weather. Keep an eye on it. 🌧️",
            "curious": "Your cat is curious about something. Let it explore! 🕵️", "playful": "Your cat is feeling playful. Time for some fun! 🧶"
        }
        
        audio_messages = {
            'Angry': "Your cat is expressing anger or frustration. They might need some space! 😾", 'Defence': "Your cat is in a defensive mode - they might feel threatened 🛡️",
            'Fighting': "Your cat is showing aggressive behavior - best to keep distance! ⚔️", 'Happy': "Your cat is expressing joy and contentment! 😺",
            'HuntingMind': "Your cat is in hunting mode - they're feeling predatory! 🐾", 'Mating': "Your cat is making mating calls 💕",
            'MotherCall': "Your cat is making nurturing sounds, typical of mother cats! 🤱", 'Paining': "Your cat might be in pain or distress - consider a vet visit! 🏥",
            'Resting': "Your cat is making peaceful, relaxed sounds 😴", 'Warning': "Your cat is trying to warn about something - they might feel unsafe! ⚠️"
        }
        
        base_message = f"I think your cat is feeling {emotion}. "
        health_message = "They appear to be healthy! 🌟" if sickness == "normal" else "They might not be feeling well - consider a check-up! 🏥"
        
        if mood:
            mood_message = f"\nBased on their vocalizations: {audio_messages.get(mood, f'They seem to be in a {mood.lower()} state.')}"

        return base_message + health_message

    def process_image(image_path):
        try:
            image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            emotion_predictions = image_model.predict(img_array, verbose=0)
            predicted_emotion = IMAGE_CLASSES[np.argmax(emotion_predictions)]
            confidence = np.max(emotion_predictions)

            sickness_predictions = sickness_model.predict(img_array, verbose=0)
            predicted_sickness = SICKNESS_CLASSES[np.argmax(sickness_predictions)]

            return predicted_emotion, predicted_sickness, confidence
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            return None, None, None

    def extract_audio(video_path):
        try:
            temp_audio = os.path.join(os.path.dirname(video_path), "temp_audio.wav")
            command = f"ffmpeg -i \"{video_path}\" -q:a 0 -map a \"{temp_audio}\" -y"
            subprocess.run(command, shell=True, check=True)
            return temp_audio
        except Exception as e:
            messagebox.showerror("Error", f"Audio extraction error: {str(e)}")
            return None

    def process_audio(audio_path):
        try:
            y, sr = librosa.load(audio_path, duration=3)
            mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
            mel_spect = cv2.resize(mel_spect, (128, 128))
            mel_spect = np.expand_dims(mel_spect, axis=-1)
            mel_spect = np.expand_dims(mel_spect, axis=0)

            audio_predictions = audio_model.predict(mel_spect, verbose=0)
            predicted_mood = AUDIO_CLASSES[np.argmax(audio_predictions)]
            
            return predicted_mood
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process audio: {str(e)}")
            return None

    def process_video(video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Could not read video frame")
                return None, None, None, None

            temp_frame = os.path.join(os.path.dirname(video_path), "temp_frame.jpg")
            cv2.imwrite(temp_frame, frame)
            
            emotion, sickness, confidence = process_image(temp_frame)
            if os.path.exists(temp_frame):
                os.remove(temp_frame)
            cap.release()

            audio_mood = None
            temp_audio = extract_audio(video_path)
            if temp_audio:
                audio_mood = process_audio(temp_audio)
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)

            if emotion and sickness:
                return emotion, sickness, confidence, audio_mood
            return None, None, None, None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process video: {str(e)}")
            return None, None, None, None

    def process_file():
        file_path = filedialog.askopenfilename(
            title="Select Your Cat Media",
            filetypes=[("Media Files", "*.png;*.jpg;*.jpeg;*.mp4;*.avi;*.mov;*.wav;*.mp3")]
        )
        
        if not file_path:
            return

        loading_label.config(text="Processing... Please wait! 🐱")
        root.update()

        try:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                emotion, sickness, confidence = process_image(file_path)
                if emotion and sickness:
                    friendly_message = get_friendly_message(emotion, sickness, confidence)
                    result_label.config(text=friendly_message, wraplength=500)
                    audio_result_label.config(text="No audio analysis available for images")
                    
                    image = Image.open(file_path)
                    image.thumbnail((300, 300))
                    photo = ImageTk.PhotoImage(image)
                    screenshot_label.config(image=photo)
                    screenshot_label.image = photo
                    
            elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                emotion, sickness, confidence, audio_mood = process_video(file_path)
                if emotion and sickness:
                    friendly_message = get_friendly_message(emotion, sickness, confidence, audio_mood)
                    result_label.config(text=friendly_message, wraplength=500)

                    if audio_mood:
                        audio_result_label.config(text=f"Audio Analysis: {audio_messages.get(audio_mood)} 🔊", fg='#333333')
                    else:
                        audio_result_label.config(text="Audio analysis unavailable. Please ensure your video has audio.", fg='#666666')

                    cap = cv2.VideoCapture(file_path)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame_rgb)
                        image.thumbnail((300, 300))
                        photo = ImageTk.PhotoImage(image)
                        screenshot_label.config(image=photo)
                        screenshot_label.image = photo
                    cap.release()

            elif file_path.lower().endswith(('.wav', '.mp3')):
                audio_mood = process_audio(file_path)
                if audio_mood:
                    result_label.config(text="Audio Analysis Only", wraplength=500)
                    audio_result_label.config(text=f"{audio_messages.get(audio_mood)} 🔊", fg='#333333')
                    screenshot_label.config(image='')

            loading_label.config(text="")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            loading_label.config(text="")

    root = tk.Tk()
    root.title("🐱 Cat Mood & Health Analyzer")
    root.geometry("800x800")
    root.configure(bg='#f0f0f0')

    style = ttk.Style()
    style.configure('Custom.TButton', font=('Arial', 12))

    header = tk.Label(root, text="Cat Mood & Health Analyzer", font=("Arial", 24, "bold"), bg='#f0f0f0', fg='#333333')
    header.pack(pady=30)

    description = tk.Label(root, text="Upload a photo, video, or audio file of your cat to analyze their mood and health status!", font=("Arial", 12), bg='#f0f0f0', fg='#666666', wraplength=600)
    description.pack(pady=10)

    process_button = ttk.Button(root, text="Upload Cat Media 📁", command=process_file, style='Custom.TButton')
    process_button.pack(pady=20)

    loading_label = tk.Label(root, text="", font=("Arial", 12), bg='#f0f0f0', fg='#666666')
    loading_label.pack(pady=10)

    result_frame = tk.Frame(root, bg='#f0f0f0')
    result_frame.pack(pady=20, padx=50, fill='both', expand=True)

    result_label = tk.Label(result_frame, text="Your cat's analysis will appear here! 😺", font=("Arial", 14), bg='#ffffff', fg='#333333', wraplength=500, pady=20, padx=20, relief='ridge', borderwidth=1)
    result_label.pack(pady=20)

    audio_result_label = tk.Label(result_frame, text="Audio analysis will appear here for videos and audio files! 🔊", font=("Arial", 14), bg='#ffffff', fg='#666666', wraplength=500, pady=20, padx=20, relief='ridge', borderwidth=1)
    audio_result_label.pack(pady=20)

    screenshot_label = tk.Label(root, bg='#f0f0f0')
    screenshot_label.pack(pady=20)

    footer = tk.Label(root, text="Made with ❤️ for cats everywhere", font=("Arial", 10), bg='#f0f0f0', fg='#999999')
    footer.pack(pady=20)

    root.mainloop()


---

8. Conclusion
=============

Le projet **Cat Mood & Health Analyzer** représente une avancée intéressante dans l'analyse du bien-être des animaux de compagnie, en particulier les chats. Grâce à l'intelligence artificielle et aux modèles d'apprentissage profond, il offre une méthode automatisée pour évaluer l'état émotionnel et de santé des chats à partir de simples fichiers multimédia. 

À l'avenir, nous envisageons d'élargir les capacités de cette application en ajoutant de nouvelles fonctionnalités, telles que la détection d'autres maladies ou comportements spécifiques, et en affinant les modèles pour augmenter leur précision. Nous souhaitons également intégrer des fonctionnalités permettant aux vétérinaires et propriétaires d'animaux de suivre l'évolution de la santé des chats au fil du temps.

L'application pourrait trouver des applications dans diverses industries, telles que la santé animale, les refuges pour animaux, et même dans les maisons pour surveiller le bien-être des animaux domestiques. En simplifiant l'analyse du comportement des chats, ce projet pourrait offrir un outil précieux pour améliorer la qualité de vie de nos compagnons félins.


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

