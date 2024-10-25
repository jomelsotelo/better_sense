import threading
import tensorflow as tf
import numpy as np
import io
import csv
import sounddevice as sd
from scipy.signal import correlate

# Function to define sound classification
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]  # Skip CSV header
    return class_names

# Download the model to YAMNet.tflite and Default setup
interpreter = tf.lite.Interpreter('lite-model_yamnet_tflite_1.tflite')
input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']
embeddings_output_index = output_details[1]['index']
spectrogram_output_index = output_details[2]['index']

# Download the YAMNet class map to yamnet_class_map.csv
class_names = class_names_from_csv(open('yamnet_class_map.csv').read())

# Global variables
scores = None
top_N = 3  # Top N results for classification

# Set the sample rate for audio
SAMPLE_RATE = 48000  # Standard sample rate for audio
CHANNELS = 2  # Stereo input (left and right)

# Callback function for processing real-time audio stream
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    
    # Separate the left and right channels
    left_channel = indata[:, 0]  # Left channel data
    right_channel = indata[:, 1]  # Right channel data

    # Left-Right Detection
    correlation = correlate(left_channel, right_channel, mode="full")
    time_diff = np.argmax(correlation) - len(left_channel)
    amplitude_diff = np.mean(np.abs(left_channel)) - np.mean(np.abs(right_channel))
    
    if time_diff > 0 and amplitude_diff > 0:
        print("Sound is coming from the left")
    elif time_diff < 0 and amplitude_diff < 0:
        print("Sound is coming from the right")
    else:
        print("Sound direction is unclear")
    
    # Audio classification using YAMNet
    global interpreter, waveform_input_index, scores_output_index, scores
    # Convert the audio input to the format YAMNet expects (mono, 16kHz, float32)
    mono_audio = np.mean(indata, axis=1)  # Convert stereo to mono
    resampled_audio = np.interp(np.arange(0, len(mono_audio), len(mono_audio) / 16000), np.arange(0, len(mono_audio)), mono_audio)
    waveform = resampled_audio.astype(np.float32) / 32768.0  # Normalize

    interpreter.resize_tensor_input(waveform_input_index, [len(waveform)], strict=True)
    interpreter.allocate_tensors()
    interpreter.set_tensor(waveform_input_index, waveform)
    interpreter.invoke()

    scores = interpreter.get_tensor(scores_output_index)

# Function to get top N sound classification results
def update_result_text():
    global scores, top_N, class_names

    if scores is not None:
        result = np.mean(scores, axis=0)
        top_indices = np.argsort(result)[::-1][:top_N]
        top_results = [(class_names[i], result[i]) for i in top_indices]

        print("\nWhat is happening around me:")
        for class_name, score in top_results:
            print(f"{class_name}: {score:.3f}")

# Function to start audio stream
def start_audio_stream():
    print(sd.query_devices())
    device_index = int(input("Enter the device index for your microphone: "))
    
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
                update_result_text()  # Periodically update classification results
    except KeyboardInterrupt:
        print("\nStopping the audio stream.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Start the audio processing
start_audio_stream()