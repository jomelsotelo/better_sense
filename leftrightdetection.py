import sounddevice as sd
import numpy as np
from scipy.signal import correlate

# Set the sample rate for audio
SAMPLE_RATE = 48000  # Standard sample rate for audio
CHANNELS = 2  # Stereo input (left and right)

# List all available audio devices to choose the desired one
print("Available audio devices:")
print(sd.query_devices())

# Specify the device index for the microphone you want to use
device_index = int(input("Enter the device index for your microphone: "))

# Callback function to process real-time audio stream
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")

    # Separate the left and right channels
    left_channel = indata[:, 0]  # Left channel data
    right_channel = indata[:, 1]  # Right channel data

    # Compute Time Difference of Arrival (TDOA) using cross-correlation
    correlation = correlate(left_channel, right_channel, mode="full")
    time_diff = np.argmax(correlation) - len(left_channel)

    # Compute amplitude difference
    amplitude_diff = np.mean(np.abs(left_channel)) - np.mean(np.abs(right_channel))

    # Determine sound direction based on TDOA and amplitude differences
    if time_diff > 0 and amplitude_diff > 0:
        print("Sound is coming from the left")
    elif time_diff < 0 and amplitude_diff < 0:
        print("Sound is coming from the right")
    else:
        print("Sound direction is unclear")

# Set up the input stream with the chosen device
try:
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        device=device_index,
        callback=audio_callback
    )

    # Start the audio stream
    print("Starting the audio stream... Press Ctrl+C to stop.")
    with stream:
        while True:
            pass  # Keep the stream running
except KeyboardInterrupt:
    print("\nStopping the audio stream.")
except Exception as e:
    print(f"An error occurred: {e}")
