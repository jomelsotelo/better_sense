import streamlit as st
import cv2
import numpy as np
import sounddevice as sd
from scipy.signal import correlate
import threading

# Set the sample rate for audio
SAMPLE_RATE = 48000  # Standard sample rate for audio
CHANNELS = 2  # Stereo input (left and right)

# Streamlit UI setup
st.title("🌌 Better Sense")
st.write(
    "An audio visualizer to alert individuals of their surroundings, creating a more immersive and informed experience."
)

st.header("Key Features")
st.write("Detect sound direction (left or right) and display an icon notifying users of the detected sound.")

# User inputs for camera and audio
camera_index = st.selectbox("Select Camera Index", options=[0, 1, 2], index=0, key="camera_index_select")
audio_devices = sd.query_devices()
audio_device_names = [device['name'] for device in audio_devices]
audio_device_index = st.selectbox("Select Audio Device", options=range(len(audio_device_names)), format_func=lambda x: audio_device_names[x], key="audio_device_select")

run_camera = st.checkbox("Show Local Camera Feed", key="camera_feed_checkbox")

# Placeholder for the video feed
stframe = st.empty()

# Shared variable to hold sound direction
sound_direction = "none"

def audio_callback(indata, frames, time, status):
    global sound_direction
    if status:
        print(f"Status: {status}")

    # Separate the left and right channels
    left_channel = indata[:, 0]  # Left channel data
    right_channel = indata[:, 1]  # Right channel data

    # Compute amplitude in dB for each channel
    left_amplitude_db = 20 * np.log10(np.mean(np.abs(left_channel)) + 1e-6)  # Adding small value to avoid log(0)
    right_amplitude_db = 20 * np.log10(np.mean(np.abs(right_channel)) + 1e-6)

    # Set threshold to -10 dB
    threshold_db = -30

    # Determine if each channel exceeds the threshold
    left_activity = left_amplitude_db > threshold_db
    right_activity = right_amplitude_db > threshold_db

    # Determine sound direction based on dB threshold and TDOA
    if left_activity or right_activity:
        # Compute Time Difference of Arrival (TDOA) using cross-correlation
        correlation = correlate(left_channel, right_channel, mode="full")
        time_diff = np.argmax(correlation) - len(left_channel)

        # Compute amplitude difference
        amplitude_diff = np.mean(np.abs(left_channel)) - np.mean(np.abs(right_channel))

        # Update sound direction
        if left_activity and time_diff > 0 and amplitude_diff > 0:
            sound_direction = "left"
        elif right_activity and time_diff < 0 and amplitude_diff < 0:
            sound_direction = "right"
        else:
            sound_direction = "none"
    else:
        sound_direction = "none"

def start_audio_stream():
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            device=audio_device_index,
            callback=audio_callback
        )
        with stream:
            while run_camera:
                pass  # Keep the stream running
    except Exception as e:
        st.error(f"An error occurred with the audio stream: {e}")

if run_camera:
    # Start audio stream in a separate thread
    audio_thread = threading.Thread(target=start_audio_stream, daemon=True)
    audio_thread.start()

    # Open the selected camera
    cap = cv2.VideoCapture(camera_index)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video frame")
            break

        # Overlay text indicators on the video feed
        height, width, _ = frame.shape
        if sound_direction == "left":
            cv2.putText(frame, "Left", (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        elif sound_direction == "right":
            cv2.putText(frame, "Right", (width - 150, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the video frame with the text overlay
        stframe.image(frame, channels="BGR")

        # Exit loop if the checkbox is unchecked
        if not st.session_state.get("camera_feed_checkbox"):
            break

    cap.release()
    cv2.destroyAllWindows()
