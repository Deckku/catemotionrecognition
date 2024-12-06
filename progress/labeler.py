import streamlit as st
import os
import random
import shutil
from PIL import Image

# Setup directories (your source and destination folders)
source_folder = "C:\\Users\\adnan\\Desktop\\IA\\projectmchach\\9kcats\\train"
dest_folder = "C:\\Users\\adnan\\Desktop\\IA\\projectmchach\\gooddataset\\train"

# Initialize session state if not already done
if "label" not in st.session_state:
    st.session_state.label = None  # Placeholder for selected label

# Get list of image files in source folder
image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]

# Display image selection and labeling
if len(image_files) > 0:
    try:
        # Display the current image
        current_image = random.choice(image_files)
        image_path = os.path.join(source_folder, current_image)
        image = Image.open(image_path)
        st.image(image, caption=f"Current Image: {current_image}")

        # Create a grid of buttons for labels
        labels = ['angry', 'beg', 'disgusted', 'frightened', 'happy', 'normal', 'sad', 'scared', 'sick', 'surprised', 'wonder']

        # Create columns for buttons
        col1, col2, col3 = st.columns(3)  # You can adjust the number of columns as needed
        
        # Define button actions in each column
        with col1:
            if st.button(labels[0]):
                selected_label = labels[0]
                st.session_state.label = selected_label
        with col2:
            if st.button(labels[1]):
                selected_label = labels[1]
                st.session_state.label = selected_label
        with col3:
            if st.button(labels[2]):
                selected_label = labels[2]
                st.session_state.label = selected_label
        with col1:
            if st.button(labels[3]):
                selected_label = labels[3]
                st.session_state.label = selected_label
        with col2:
            if st.button(labels[4]):
                selected_label = labels[4]
                st.session_state.label = selected_label
        with col3:
            if st.button(labels[5]):
                selected_label = labels[5]
                st.session_state.label = selected_label
        with col1:
            if st.button(labels[6]):
                selected_label = labels[6]
                st.session_state.label = selected_label
        with col2:
            if st.button(labels[7]):
                selected_label = labels[7]
                st.session_state.label = selected_label
        with col3:
            if st.button(labels[8]):
                selected_label = labels[8]
                st.session_state.label = selected_label
        with col1:
            if st.button(labels[9]):
                selected_label = labels[9]
                st.session_state.label = selected_label
        with col2:
            if st.button(labels[10]):
                selected_label = labels[10]
                st.session_state.label = selected_label

        # When a button is clicked, move the image to the corresponding folder
        if st.session_state.label is not None:
            label_folder = os.path.join(dest_folder, selected_label)
            
            # Create the folder if it doesn't exist
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)
            
            # Move the image to the corresponding folder
            shutil.move(image_path, os.path.join(label_folder, current_image))
            
            st.success(f"Image labeled as {selected_label} and moved to {label_folder}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("No images to label.")
