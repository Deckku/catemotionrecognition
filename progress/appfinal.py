import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import cv2
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageTk
import random
import os
import subprocess

# Constants
IMG_HEIGHT, IMG_WIDTH = 128, 128
IMAGE_CLASSES = ["angry", "beg", "annoyed", "frightened", "happy", "normal", "sad", "scared", "under the weather", "curious", "playful"]
SICKNESS_CLASSES = ["normal", "sick"]
AUDIO_CLASSES = ['Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind', 'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning']

# Load trained models
try:
    image_model = load_model('catemotionrecognition\\progress\\maybe3.h5')
    sickness_model = load_model('catemotionrecognition\\progress\\sicknormalf.h5')
    audio_model = load_model('catemotionrecognition\\progress\\issa.h5')
except:
    messagebox.showerror("Error", "Could not load one or more models. Please check if model files exist.")
    exit()

def get_friendly_message(emotion, sickness, confidence, mood=None):
    if confidence < 0.1:  # 60% confidence threshold
        return "I'm not confident this is a cat image, or the image might be unclear. Please try uploading a clearer picture of a cat! 🐱"
    
    messages = {
        "happy": "Your cat seems to be in a great mood! 😊",
        "sad": "Aww, your cat might need some extra love and attention right now 💕",
        "angry": "Looks like someone woke up on the wrong side of the bed! 😾",
        "sick": "Your cat might not be feeling well. Consider a vet visit! 🏥",
        "normal": "Your cat appears to be healthy! 🌟",
        "beg": "Your cat is begging for something. Maybe it's time for a treat! 🍖",
        "annoyed": "Your cat seems annoyed. It might need some space. 😒",
        "frightened": "Your cat seems frightened. Try to comfort it. 😨",
        "scared": "Your cat is scared. It might need some reassurance. 😱",
        "under the weather": "Your cat seems under the weather. Keep an eye on it. 🌧️",
        "curious": "Your cat is curious about something. Let it explore! 🕵️",
        "playful": "Your cat is feeling playful. Time for some fun! 🧶"
    }
    
    audio_messages = {
        'Angry': "Your cat is expressing anger or frustration. They might need some space! 😾",
        'Defence': "Your cat is in a defensive mode - they might feel threatened 🛡️",
        'Fighting': "Your cat is showing aggressive behavior - best to keep distance! ⚔️",
        'Happy': "Your cat is expressing joy and contentment! 😺",
        'HuntingMind': "Your cat is in hunting mode - they're feeling predatory! 🐾",
        'Mating': "Your cat is making mating calls 💕",
        'MotherCall': "Your cat is making nurturing sounds, typical of mother cats! 🤱",
        'Paining': "Your cat might be in pain or distress - consider a vet visit! 🏥",
        'Resting': "Your cat is making peaceful, relaxed sounds 😴",
        'Warning': "Your cat is trying to warn about something - they might feel unsafe! ⚠️"
    }
    
    base_message = f"I think your cat is feeling {emotion}. "
    health_message = "They appear to be healthy! 🌟" if sickness == "normal" else "They might not be feeling well - consider a check-up! 🏥"
    
    if mood:
        mood_message = f"\nBased on their vocalizations: {audio_messages.get(mood, f'They seem to be in a {mood.lower()} state.')}"
        return base_message + health_message 
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
        if len(y) == 0:
            messagebox.showerror("Error", "No audio data found in the file")
            return None
            
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

def process_file():
    file_path = filedialog.askopenfilename(
        title="Select Your Cat Media",
        filetypes=[
            ("Media Files", "*.png;*.jpg;*.jpeg;*.mp4;*.avi;*.mov;*.wav;*.mp3")
        ]
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
                result_label.config(
                    text=friendly_message,
                    wraplength=500
                )
                audio_result_label.config(text="No audio analysis available for images")
                
                # Display the processed image
                image = Image.open(file_path)
                image.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(image)
                screenshot_label.config(image=photo)
                screenshot_label.image = photo
                
        elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            emotion, sickness, confidence, audio_mood = process_video(file_path)
            if emotion and sickness:
                friendly_message = get_friendly_message(emotion, sickness, confidence, audio_mood)
                result_label.config(
                    text=friendly_message,
                    wraplength=500
                )
                
                # Update audio analysis
                if audio_mood:
                    audio_messages = {
                        'Angry': "Your cat is expressing anger or frustration. They might need some space! 😾",
                        'Defence': "Your cat is in a defensive mode - they might feel threatened 🛡️",
                        'Fighting': "Your cat is showing aggressive behavior - best to keep distance! ⚔️",
                        'Happy': "Your cat is expressing joy and contentment! 😺",
                        'HuntingMind': "Your cat is in hunting mode - they're feeling predatory! 🐾",
                        'Mating': "Your cat is making mating calls 💕",
                        'MotherCall': "Your cat is making nurturing sounds, typical of mother cats! 🤱",
                        'Paining': "Your cat might be in pain or distress - consider a vet visit! 🏥",
                        'Resting': "Your cat is making peaceful, relaxed sounds 😴",
                        'Warning': "Your cat is trying to warn about something - they might feel unsafe! ⚠️"
                    }
                    audio_result_label.config(
                        text=f"Audio Analysis: {audio_messages.get(audio_mood, f'Your cat is in a {audio_mood.lower()} state')} 🔊",
                        fg='#333333'
                    )
                else:
                    audio_result_label.config(
                        text="Audio analysis unavailable. Please ensure your video has audio.",
                        fg='#666666'
                    )
                
                # Display first frame of video
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
                result_label.config(
                    text="Audio Analysis Only",
                    wraplength=500
                )
                audio_messages = {
                    'Angry': "Your cat is expressing anger or frustration. They might need some space! 😾",
                    'Defence': "Your cat is in a defensive mode - they might feel threatened 🛡️",
                    'Fighting': "Your cat is showing aggressive behavior - best to keep distance! ⚔️",
                    'Happy': "Your cat is expressing joy and contentment! 😺",
                    'HuntingMind': "Your cat is in hunting mode - they're feeling predatory! 🐾",
                    'Mating': "Your cat is making mating calls 💕",
                    'MotherCall': "Your cat is making nurturing sounds, typical of mother cats! 🤱",
                    'Paining': "Your cat might be in pain or distress - consider a vet visit! 🏥",
                    'Resting': "Your cat is making peaceful, relaxed sounds 😴",
                    'Warning': "Your cat is trying to warn about something - they might feel unsafe! ⚠️"
                }
                audio_result_label.config(
                    text=f"{audio_messages.get(audio_mood, f'Your cat is in a {audio_mood.lower()} state')} 🔊",
                    fg='#333333'
                )
                screenshot_label.config(image='')
            
        loading_label.config(text="")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        loading_label.config(text="")

# Create a modern-looking GUI
root = tk.Tk()
root.title("🐱 Cat Mood & Health Analyzer")
root.geometry("800x800")
root.configure(bg='#f0f0f0')

style = ttk.Style()
style.configure('Custom.TButton', font=('Arial', 12))

header = tk.Label(
    root,
    text="Cat Mood & Health Analyzer",
    font=("Arial", 24, "bold"),
    bg='#f0f0f0',
    fg='#333333'
)
header.pack(pady=30)

description = tk.Label(
    root,
    text="Upload a photo, video, or audio file of your cat to analyze their mood and health status!",
    font=("Arial", 12),
    bg='#f0f0f0',
    fg='#666666',
    wraplength=600
)
description.pack(pady=10)

process_button = ttk.Button(
    root,
    text="Upload Cat Media 📁",
    command=process_file,
    style='Custom.TButton'
)
process_button.pack(pady=20)

loading_label = tk.Label(
    root,
    text="",
    font=("Arial", 12),
    bg='#f0f0f0',
    fg='#666666'
)
loading_label.pack(pady=10)

result_frame = tk.Frame(root, bg='#f0f0f0')
result_frame.pack(pady=20, padx=50, fill='both', expand=True)

result_label = tk.Label(
    result_frame,
    text="Your cat's analysis will appear here! 😺",
    font=("Arial", 14),
    bg='#ffffff',
    fg='#333333',
    wraplength=500,
    pady=20,
    padx=20,
    relief='ridge',
    borderwidth=1
)
result_label.pack(pady=20)

audio_result_label = tk.Label(
    result_frame,
    text="Audio analysis will appear here for videos and audio files! 🔊",
    font=("Arial", 14),
    bg='#ffffff',
    fg='#666666',
    wraplength=500,
    pady=20,
    padx=20,
    relief='ridge',
    borderwidth=1
)
audio_result_label.pack(pady=20)

screenshot_label = tk.Label(root, bg='#f0f0f0')
screenshot_label.pack(pady=20)

footer = tk.Label(
    root,
    text="Made with ❤️ for cats everywhere",
    font=("Arial", 10),
    bg='#f0f0f0',
    fg='#999999'
)
footer.pack(pady=20)

root.mainloop()

# Export the app as an executable
import PyInstaller.__main__

PyInstaller.__main__.run([
    '--name=%s' % "CatCompanion",
    '--onefile',
    '--windowed',
    'c:\\Users\\adnan\\Desktop\\IA\\cursor\\import tkinter as tk.py'
])
