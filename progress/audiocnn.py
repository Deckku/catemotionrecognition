import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import os
import tempfile

# Définir les classes d'émotions
CLASSES = ['Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind', 
           'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning']

# Charger le modèle
@st.cache_resource
def load_model(uploaded_file):
    # Sauvegarder temporairement le fichier .h5 ou .hdf5
    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    try:
        # Charger le modèle depuis le fichier temporaire
        # Utilisation de compile=False pour éviter les incompatibilités de compilation
        model = tf.keras.models.load_model(temp_file_path, compile=False)
    except Exception as e:
        # Afficher une erreur détaillée si le chargement échoue
        raise ValueError(f"Impossible de charger le modèle : {e}")
    finally:
        # Nettoyer le fichier temporaire
        os.remove(temp_file_path)

    return model

# Prétraitement de l'audio : conversion en spectrogramme log-mel
def preprocess_audio(file_path, target_shape=(128, 128)):
    # Charger l'audio
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')

    # Générer le spectrogramme log-mel
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=target_shape[0])
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Ajuster la taille pour correspondre à target_shape
    if log_spectrogram.shape[1] < target_shape[1]:
        pad_width = target_shape[1] - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        log_spectrogram = log_spectrogram[:, :target_shape[1]]

    # Ajouter des dimensions pour correspondre à l'entrée du modèle (H, W, C)
    return log_spectrogram[np.newaxis, ..., np.newaxis]

# Interface utilisateur Streamlit
st.title("Cat Emotion Recognition")

st.write("""
Téléchargez un fichier audio de chat pour prédire son émotion. 
Le modèle supporte les classes suivantes :
""")
st.write(", ".join(CLASSES))

# Importation du modèle
uploaded_model = st.file_uploader("Téléchargez un fichier modèle (.h5 ou .hdf5 uniquement)", type=["h5", "hdf5"])
model = None

if uploaded_model is not None:
    try:
        model = load_model(uploaded_model)
        st.success("Modèle chargé avec succès!")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")

# Chargement de l'audio
audio_file = st.file_uploader("Téléchargez un fichier audio (format WAV ou MP3)", type=["wav", "mp3"])

if audio_file is not None and model is not None:
    # Sauvegarder temporairement le fichier audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_file.getvalue())
        temp_audio_path = temp_audio_file.name

    # Prétraiter l'audio
    try:
        target_shape = (128, 128)  # Ajustez selon votre modèle
        audio_input = preprocess_audio(temp_audio_path, target_shape=target_shape)
        st.success("Prétraitement terminé!")

        # Prédire l'émotion
        predictions = model.predict(audio_input)
        predicted_class = CLASSES[np.argmax(predictions)]
        st.write(f"L'émotion prédite est : *{predicted_class}*")

        # Afficher les probabilités sous forme de graphique
        st.bar_chart(predictions.flatten())
    except Exception as e:
        st.error(f"Erreur dans le traitement de l'audio : {e}")
    finally:
        # Nettoyer le fichier temporaire
        os.remove(temp_audio_path)
else:
    if audio_file is None:
        st.warning("Veuillez télécharger un fichier audio.")
    if model is None:
        st.warning("Veuillez importer un modèle valide.")