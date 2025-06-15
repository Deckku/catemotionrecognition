# Cat Mood & Health Analyzer

## Overview
The **Cat Mood & Health Analyzer** is an innovative project aimed at enhancing the well-being of domestic cats through artificial intelligence and machine learning. This application leverages multimodal data (images, audio, and video) to diagnose a cat's emotional state and health condition in real-time. By combining three specialized machine learning models, the project provides a comprehensive analysis of a cat's mood and health, deployed on user-friendly platforms like Streamlit and a desktop application built with Tkinter.

## Project Goals
- Develop machine learning models to:
  - Recognize cat emotions through vocalizations.
  - Identify cat emotions via images.
  - Assess cat health status from images.
- Combine these models to analyze real-time video footage for a holistic understanding of a cat’s state.
- Deploy the solution as an accessible application for pet owners to monitor their cats' well-being.

## Features
- **Image Analysis**: Upload a clear image of your cat to predict its emotional state (e.g., happy, sad, angry, curious) and health status (normal or sick).
- **Video Analysis**: Process video files to analyze both visual and audio components, providing a comprehensive report on the cat’s emotions and health.
- **Audio Analysis**: Upload audio recordings of cat vocalizations (e.g., meows, purrs) to detect moods such as anger, happiness, or pain.
- **User-Friendly Interface**: A simple, intuitive desktop application built with Tkinter, suitable for cat owners and professionals alike.
- **Real-Time Feedback**: Immediate results with easy-to-understand messages about your cat’s emotional and health status.

## Models
The project integrates three machine learning models:
1. **Emotion Recognition (Audio)**: A convolutional neural network (CNN) trained on 5,938 audio files across 10 emotional classes (e.g., Angry, Happy, Paining). Audio is preprocessed into spectrograms for analysis.
2. **Emotion Recognition (Image)**: A CNN model trained on diverse cat images to classify emotions into 11 categories (e.g., Happy, Sad, Playful, Curious).
3. **Health Status Recognition (Image)**: A CNN model that distinguishes between healthy and sick cats based on visual cues in images.

These models are combined to process video frames and audio, providing dynamic insights into a cat’s state over time.

## Data
- **Audio Data**: A dataset of 5,938 audio files across 10 emotional classes, sourced from Yagya Raj Pandeya, with a separate 100-file test set from Kaggle.
- **Image Data**: Collected from public sources and specialized databases, ensuring diversity in cat expressions and health conditions. Images are preprocessed (resized, normalized) for model training.
- **Video Data**: Videos are processed by extracting frames for image-based analysis and audio for vocalization analysis.

## Deployment
The application is deployed in two formats:
- **Streamlit**: A web-based interface for easy access and real-time analysis.
- **Desktop Application**: A Tkinter-based GUI that allows users to upload media files (images, videos, audio) and view results. The app can be packaged as a standalone executable using PyInstaller.

## Getting Started
1. Launch the desktop application or access the Streamlit platform.
2. Upload an image, video, or audio file of your cat.
3. View the detailed analysis of your cat’s mood, health, and vocalizations.
4. Use the insights to better care for your feline companion.

## Future Work
- Enhance model accuracy with larger, more diverse datasets.
- Add detection for specific diseases or behaviors.
- Enable longitudinal tracking of a cat’s health and emotional trends.
- Explore integration with veterinary platforms for professional use.

## Applications
- **Pet Owners**: Monitor cats’ well-being at home.
- **Veterinarians**: Use as a diagnostic aid.
- **Animal Shelters**: Assess the health and emotional state of cats for better care and adoption matching.
- **Research**: Study feline behavior and health patterns.

## Conclusion
The Cat Mood & Health Analyzer is a pioneering tool that combines AI with a passion for feline welfare. By automating the analysis of cat emotions and health, it empowers owners and professionals to ensure their cats lead happier, healthier lives.

---

*Made with ❤️ for cats everywhere.*
