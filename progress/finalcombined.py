import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageTk
from moviepy.editor import VideoFileClip
import random

# Constants
IMG_HEIGHT, IMG_WIDTH = 128, 128
IMAGE_CLASSES = ["angry", "beg", "annoyed", "frightened", "happy", "normal", "sad", "scared", "sick", "curious", "playful"]
SICKNESS_CLASSES = ["normal", "sick"]
AUDIO_CLASSES = ['Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind', 'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning']

# Load trained models
image_model = load_model('maybe3.h5')
sickness_model = load_model('sicknormal.h5')
audio_model = load_model('issa.h5')

# Function to process an image
def process_image(image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict emotion and calculate confidence
    emotion_predictions = image_model.predict(img_array)
    predicted_emotion = IMAGE_CLASSES[np.argmax(emotion_predictions)]
    emotion_confidence = np.max(emotion_predictions) * 100  # Confidence in percentage

    # Predict sickness and calculate confidence
    sickness_predictions = sickness_model.predict(img_array)
    sickness_confidence = np.max(sickness_predictions) * 100  # Confidence in percentage
    predicted_sickness = "sick" if sickness_confidence > 50 else "normal"
    
    return predicted_emotion, emotion_confidence, predicted_sickness, sickness_confidence


# Function to process and classify images in a video based on its actual duration
def process_video_for_image_emotions(video_path):
    cap = cv2.VideoCapture(video_path)
    image_preds = []
    
    # Get total duration of the video in seconds
    video = VideoFileClip(video_path)
    total_duration = video.duration  # Duration in seconds

    for i in range(int(total_duration)):  # Iterate through each second
        current_time = i
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)  # Convert seconds to milliseconds
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess frame for image model
        resized_frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(resized_frame) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict emotion from frame
        predictions = image_model.predict(img_array)
        predicted_class = IMAGE_CLASSES[np.argmax(predictions)]
        image_preds.append(predicted_class)
    
    cap.release()

    # Determine the dominant emotion
    dominant_emotion = max(set(image_preds), key=image_preds.count)
    return dominant_emotion

# Function to select and display a random frame from the video
def display_random_screenshot(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Select a random frame index
    random_frame_index = random.randint(0, total_frames - 1)
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert the frame to an image that tkinter can display
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((IMG_HEIGHT, IMG_WIDTH))
        photo = ImageTk.PhotoImage(image)
        
        # Display the image on the GUI
        screenshot_label.config(image=photo)
        screenshot_label.image = photo  # Keep a reference to avoid garbage collection

# Function to process audio from a video
def extract_audio_from_video(video_path, audio_path="audio.wav"):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    return audio_path

def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = mel_spectrogram[:128, :128]  # Ensure it's (128, 128)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
    return mel_spectrogram

def process_audio_for_mood(audio_path):
    features = preprocess_audio(audio_path)
    predictions = audio_model.predict(features)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = AUDIO_CLASSES[predicted_class_idx]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# Function to process sickness detection from images and include confidence level
def process_video_for_sickness(video_path):
    cap = cv2.VideoCapture(video_path)
    sickness_preds = []
    sickness_confidences = []
    
    video = VideoFileClip(video_path)
    total_duration = video.duration  # Duration in seconds
    
    for i in range(int(total_duration)):
        current_time = i
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()

        if not ret:
            break

        resized_frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(resized_frame) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = sickness_model.predict(img_array)
        predicted_class = SICKNESS_CLASSES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        sickness_preds.append(predicted_class)
        sickness_confidences.append(confidence)
    
    cap.release()

    sickness_status = max(set(sickness_preds), key=sickness_preds.count)
    avg_confidence = np.mean([conf for i, conf in enumerate(sickness_confidences) if sickness_preds[i] == sickness_status])

    return sickness_status, avg_confidence

# Function to run all models and display the results
def process_file():
    file_path = filedialog.askopenfilename(title="Select a File", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg"), ("Video Files", "*.mp4;*.avi;*.mov")])
    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return
    
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        emotion, emotion_confidence, sickness, sickness_confidence = process_image(file_path)
        result_label.config(
            text=(
                f"Emotion: {emotion} (Confidence: {emotion_confidence:.2f}%)\n"
                f"Sickness Status: {sickness} (Confidence: {sickness_confidence:.2f}%)"
            )
        )
    else:
        dominant_emotion = process_video_for_image_emotions(file_path)
        sickness_status, sickness_confidence = process_video_for_sickness(file_path)
        audio_path = extract_audio_from_video(file_path)
        predicted_mood, mood_confidence = process_audio_for_mood(audio_path)

        result_label.config(
            text=(
                f"Dominant Emotion: {dominant_emotion}\n"
                f"Sickness Status: {sickness_status} (Confidence: {sickness_confidence:.2f}%)\n"
                f"Mood: {predicted_mood} (Confidence: {mood_confidence:.2f}%)"
            )
        )
    
    display_random_screenshot(file_path)

# Initialize the Tkinter window
root = tk.Tk()
root.title("Cat Emotion, Sickness & Mood Detection")
root.geometry("600x600")

process_button = tk.Button(root, text="Upload and Process File", command=process_file)
process_button.pack(pady=20)

result_label = tk.Label(root, text="Prediction will be displayed here.", font=("Arial", 14))
result_label.pack(pady=20)

screenshot_label = tk.Label(root)
screenshot_label.pack(pady=20)

root.mainloop()
