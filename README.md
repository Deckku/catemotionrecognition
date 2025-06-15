# 🐱 Cat Companion Project
<img width="900" alt="banner" src="https://github.com/user-attachments/assets/f3cfe3a0-98ef-4055-98a4-7420454059e2" />

An AI-powered application for real-time analysis of cat emotions and health status using multimodal machine learning.

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Model Architecture](#️-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Support](#-support)

## 🎯 Overview

Cat Companion is an innovative machine learning project that combines computer vision and audio processing to analyze cat behavior and health in real-time. The system uses three specialized models to provide comprehensive insights into your cat's well-being:

1. **Emotion Recognition from Images** - Analyzes facial expressions and body language
2. **Audio Mood Detection** - Processes vocalizations like meows and purrs
3. **Health Status Assessment** - Identifies potential health issues from visual cues

## ✨ Features

### 🖼️ Image Analysis
- **Emotion Detection**: Identifies 11 different emotional states including happy, sad, angry, playful, curious, and more
- **Health Assessment**: Determines if your cat appears healthy or may need veterinary attention
- **High Accuracy**: Trained on thousands of cat images for reliable predictions

### 🎵 Audio Analysis
- **Vocalization Classification**: Recognizes 10 types of cat sounds including angry, happy, hunting mind, mother call, and warning sounds
- **Real-time Processing**: Analyzes audio from uploaded files or video soundtracks
- **Emotional Context**: Provides insights into what your cat is trying to communicate

### 🎬 Video Processing
- **Combined Analysis**: Processes both visual and audio components of video files
- **Frame-by-frame Processing**: Extracts key frames for comprehensive analysis
- **Temporal Tracking**: Monitors changes in mood and health over time

### 💻 User-Friendly Interface
- **Desktop Application**: Built with Tkinter for easy installation and use
- **Drag & Drop Support**: Simple file upload for images, videos, and audio files
- **Instant Results**: Real-time analysis with friendly, actionable feedback
- **Multi-format Support**: Accepts PNG, JPG, MP4, AVI, MOV, WAV, and MP3 files

## 📊 Dataset

### Image Dataset
- **Emotion Recognition**: Curated dataset with 11 emotion categories
- **Health Assessment**: Binary classification dataset (healthy vs. sick)
- **Quality Control**: High-resolution images with clear cat features

### Audio Dataset
- **Size**: 5,938 audio files across 10 classes
- **Categories**: Angry, Defence, Fighting, Happy, HuntingMind, Mating, MotherCall, Paining, Resting, Warning
- **Format**: MP3 files processed into mel-spectrograms
- **Split**: 80% training, 20% validation

## 🏗️ Model Architecture

### Convolutional Neural Networks (CNN)
All three models use CNN architectures optimized for their specific tasks:

- **Input Processing**: Images resized to 128x128 pixels, audio converted to mel-spectrograms
- **Feature Extraction**: Multiple convolutional layers with ReLU activation
- **Classification**: Dense layers with softmax activation for multi-class prediction
- **Optimization**: Adam optimizer with categorical crossentropy loss

### Model Performance
- **Emotion Recognition**: [Insert accuracy metrics]
- **Health Detection**: [Insert accuracy metrics]  
- **Audio Classification**: [Insert accuracy metrics]

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- TensorFlow 2.x
- OpenCV
- Librosa
- Tkinter (usually included with Python)

### Quick Start
1. Clone the repository
2. Install required dependencies
3. Download pre-trained models
4. Run the application

### Executable Version
For users who prefer not to install Python, a standalone executable is available for Windows.

## 💡 Usage

### Getting Started
1. **Launch the Application**: Run the main script or executable
2. **Upload Media**: Click "Upload Cat Media" and select your file
3. **Wait for Analysis**: The AI processes your file (usually takes a few seconds)
4. **View Results**: Get detailed insights about your cat's mood and health

### Supported File Types
- **Images**: PNG, JPG, JPEG
- **Videos**: MP4, AVI, MOV  
- **Audio**: WAV, MP3

### Interpreting Results
The application provides:
- **Emotion Classification**: Clear emotional state with confidence level
- **Health Status**: Normal or potential health concerns
- **Audio Mood**: Vocalization analysis for videos and audio files
- **Friendly Messages**: Easy-to-understand explanations and recommendations

## 📈 Results

### Emotion Classes Detected
- **Angry** 😾 - Signs of irritation or aggression
- **Happy** 😺 - Content and relaxed state
- **Sad** 😿 - Signs of depression or discomfort
- **Scared/Frightened** 😨 - Fear or anxiety responses
- **Playful** 🧶 - Energetic and ready to play
- **Curious** 🕵️ - Interested in exploring
- **Normal** 😸 - Calm and comfortable
- **Annoyed** 😒 - Mildly irritated
- **Beg** 🍖 - Seeking attention or food
- **Under the Weather** 🌧️ - Not feeling well

### Audio Mood Categories
- **Happy** - Contentment vocalizations
- **Angry** - Aggressive or frustrated sounds
- **Paining** - Distress calls
- **Warning** - Alert vocalizations
- **Resting** - Peaceful sounds
- And 5 more categories...

## 📸 Screenshots

### Main Interface
<img width="521" alt="main" src="https://github.com/user-attachments/assets/aed1da71-3cec-4c82-a0e8-eadff6380d0d" />

*The clean, user-friendly interface makes it easy to upload and analyze cat media files.*

### Image Analysis Results
<img width="519" alt="Screenshot 2025-06-15 131148" src="https://github.com/user-attachments/assets/ad555ca7-65d0-4463-a10a-c84d1939d88b" />


*Example of emotion and health analysis from a cat photo.*

### Video Processing
<img width="517" alt="video" src="https://github.com/user-attachments/assets/acd8dfb0-1562-49bd-b143-2ab984638ebe" />


*Combined visual and audio analysis from video files.*

### Audio Visualization
<img width="512" alt="spectrogram" src="https://github.com/user-attachments/assets/881d080f-b3ba-463c-af81-c1ee2b718911" />

*Mel-spectrogram visualization used for audio mood detection.*


## 🔮 Future Improvements

### Short-term Goals
- **Enhanced Accuracy**: Expand training datasets for better precision
- **More Emotions**: Add additional emotional states and behaviors
- **Real-time Video**: Live camera feed analysis
- **Mobile App**: iOS and Android versions

### Long-term Vision
- **Veterinary Integration**: Professional tools for animal healthcare
- **Multi-pet Support**: Recognition for dogs and other pets
- **Behavioral Tracking**: Long-term mood and health monitoring
- **Smart Home Integration**: IoT device compatibility

### Research Areas
- **Advanced AI Models**: Explore transformer architectures
- **Behavioral Patterns**: Temporal analysis of mood changes
- **Health Prediction**: Early disease detection capabilities
- **Cross-species Analysis**: Expand to other domestic animals

## 🤝 Contributing

We welcome contributions from the community! Whether you're interested in:

- **Data Collection**: Help us gather more diverse cat images and sounds
- **Model Improvement**: Enhance existing architectures or propose new ones

Please read our contributing guidelines and code of conduct before submitting pull requests.

## 📄 License

This project is licensed under [Insert License] - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset Contributors**: Special thanks to all who provided cat images and audio samples
- **Research Community**: Built upon years of computer vision and audio processing research
- **Open Source Libraries**: TensorFlow, OpenCV, Librosa, and other amazing tools
- **Cat Owners**: Everyone who helped test and improve the application

## 📞 Created By
- Azami Hassani Adnane/ad.azamihassani@edu.umi.ac.ma
- Lamkharbech Issa/i.lamkharbech@edu.umi.ac.ma
## 📞 Supervised By:
Pr.Taoufik Masrour

---

**Made with ❤️ for cats everywhere** 🐱

*Help us make the world a better place for our feline friends, one meow at a time.*
