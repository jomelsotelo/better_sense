import streamlit as st
import cv2
import numpy as np
import os

# Title for the app
st.title("🌌 Better Sense")
st.write(
    "An audio visualizer to alert individuals of their surrounding, creating a more immersive and informed experience."
)

st.header("Key Features")
st.write(
    "Be able to detect a sound coming from the left or right"
)
st.write(
    "Display an icon notifying the users what the sound is."
)

# Load icons from the icon folder
icon_folder = "icons"
left_icon_path = os.path.join(icon_folder, "angle-double-left.png")
right_icon_path = os.path.join(icon_folder, "angle-double-right.png")

# Load the icons using OpenCV
left_icon = cv2.imread(left_icon_path, cv2.IMREAD_UNCHANGED)
right_icon = cv2.imread(right_icon_path, cv2.IMREAD_UNCHANGED)

# Function to overlay an icon with transparency
def overlay_icon(frame, icon, x, y):
    icon_height, icon_width = icon.shape[:2]
    overlay = icon[:, :, :3]  # Color channels
    mask = icon[:, :, 3]  # Alpha channel

    roi = frame[y:y+icon_height, x:x+icon_width]

    # Ensure the icon fits within the frame boundaries
    if roi.shape[0] != icon_height or roi.shape[1] != icon_width:
        return

    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - mask / 255.0) + overlay[:, :, c] * (mask / 255.0)

    frame[y:y+icon_height, x:x+icon_width] = roi

# Adding live camera feed using OpenCV
run_camera = st.checkbox("Show Local Camera Feed", key="camera_feed_checkbox")

if run_camera:
    camera_index = st.selectbox("Select Camera Index", options=[0, 1, 2], index=0, key="camera_index_select")
    cap = cv2.VideoCapture(camera_index)  # Open the selected camera
    stframe = st.empty()  # Create a placeholder for the video feed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video frame")
            break

        # Simulate sound detection with random left or right indication
        sound_direction = np.random.choice(["left", "right", "none"], p=[0.3, 0.3, 0.4])
        
        # Overlay hologram-like indicators on the video feed
        height, width, _ = frame.shape
        if sound_direction == "left" and left_icon is not None:
            overlay_icon(frame, left_icon, 50, height // 2 - left_icon.shape[0] // 2)
        elif sound_direction == "right" and right_icon is not None:
            overlay_icon(frame, right_icon, width - right_icon.shape[1] - 50, height // 2 - right_icon.shape[0] // 2)
        
        # Display the video frame with the hologram-like overlay
        stframe.image(frame, channels="BGR")

        # Exit loop if the checkbox is unchecked
        if not st.session_state.get("camera_feed_checkbox"):
            break

    cap.release()
    cv2.destroyAllWindows()