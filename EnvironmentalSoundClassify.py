# Highly recommend to run the MicCheck.py first to get the desired microphone input index and check if it is activated
# This file displays user interfaces

import customtkinter
import threading
import tensorflow as tf
import numpy as np
import io
import csv
import pyaudio


# Function to define sound classification
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]  # Skip CSV header
    return class_names

# Audio Input microphone
RATE = 16000
RECORD_SECONDS = 2
CHUNK = 1024
audio = pyaudio.PyAudio()

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

# Global variable to hold the classification thread
classification_thread = None
# Flag to control the sound classification loop
stop_classification = False
# Global variable to hold the audio stream
audio_stream = None
# Global variable to store the last calculated scores
scores = None

# User Interface Default Setup
customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green
app = customtkinter.CTk()  # create CTk window like you do with the Tk window
app.geometry("400x240")

# Title
app.title("Environmental Sound Detection")
title = customtkinter.CTkLabel(app, text="Sounds around me")
title.pack(padx=10, pady=10)

# Top 3 results
top_N = 3

# Function to open audio stream
def open_audio_stream():

    global audio_stream
    # Set microphone input index 1 tested in the MicCheck file
    audio_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=1)

# Function to close audio stream
def close_audio_stream():

    global audio_stream

    if audio_stream is not None and audio_stream.is_active():
        audio_stream.stop_stream()
        audio_stream.close()

# Function to define sound classification
def classify_sound():
    global stop_classification, audio_stream, interpreter, waveform_input_index, scores_output_index, scores

    # While loop continues until the 'stop_classification' becomes True
    while not stop_classification:
        # Open audio stream
        open_audio_stream()

        # Divide audio into frames and store all the frames in the array list
        frames = []
        for f in range(0, int(RATE / CHUNK * RECORD_SECONDS)):  # 16000/ 1024 * 2
            # Read CHUNK_SIZE bytes of audio data from a stream called stream.
            # The stream.read(CHUNK_SIZE) operation retrieves the audio data as a binary (1 or 0) string
            data = audio_stream.read(CHUNK)
            # Convert the binary string data to an array of 16-bit integers using
            # interpret the binary data as 16-bit integers and create an array from it
            frames.append(np.frombuffer(data, dtype=np.int16))

        # Convert the frame numpy array to 1D-32 float array as it required by YAMNet:
        # Horizontally stack the frames stored in the frames list.
        convert_data = np.hstack(frames)

        # Normalize audio data
        # The maximum value of a 16-bit signed integer is 32767,
        # so dividing by 32768.0 scales the values between -1.0 and 1.0
        convert_data = convert_data.astype(np.float32, order='C') / 32768.0

        # YAMNet interpreter
        waveform = convert_data

        # YAMNet setup
        interpreter.resize_tensor_input(waveform_input_index, [len(waveform)], strict=True)
        interpreter.allocate_tensors()
        interpreter.set_tensor(waveform_input_index, waveform)
        interpreter.invoke()

        scores, embeddings, spectrogram = (
            interpreter.get_tensor(scores_output_index),
            interpreter.get_tensor(embeddings_output_index),
            interpreter.get_tensor(spectrogram_output_index))

    close_audio_stream()

# Function to update the result_text widget periodically and get the top N results
def update_result_text():
    global classification_thread, scores, top_N, class_names

    if classification_thread is not None and classification_thread.is_alive() and scores is not None:

        # Get the top N results from the scores
        result = np.mean(scores, axis=0)
        top_indices = np.argsort(result)[::-1][:top_N]
        top_results = [(class_names[i], result[i]) for i in top_indices]

        # Clear the existing content of the result_text widget
        result_text.delete(1.0, customtkinter.END)
        # Update the result_text widget with the top N class names and results
        result_text.insert(customtkinter.END, "What is happening around me:\n\n".format(top_N))

        for class_name, score in top_results:
            result_text.insert(customtkinter.END, "{:12s}: {:.3f}\n".format(class_name, score))

    # Update the result_text widget again after 1000 milliseconds (1 second)
    app.after(100, update_result_text)

def button_activate():
    print("Start analyzing...")

    global classification_thread, stop_classification

    # Reset the flag
    stop_classification = False

    # Create a thread for the sound classification
    classification_thread = threading.Thread(target=classify_sound, daemon=True)
    classification_thread.start()

    # Start the update_result_text function to update the result_text widget periodically
    update_result_text()

    # Start the audio stream
    open_audio_stream()

def button_stop():
    print("Analyzing Stopped")

    global classification_thread, stop_classification

    stop_classification = True

    if classification_thread is not None and classification_thread.is_alive():
        classification_thread.join()  # Wait for the classification thread to finish

    # Stop and close the audio stream
    close_audio_stream()

# Use CTkButton instead of tkinter Button
button_start = customtkinter.CTkButton(master=app, text="Start", command=button_activate)
button_start.place(relx=0.25, rely=0.3, anchor=customtkinter.CENTER)

button_stop = customtkinter.CTkButton(master=app, text="Stop", command=button_stop, fg_color="red")
button_stop.place(relx=0.75, rely=0.3, anchor=customtkinter.CENTER)

# Create a text widget to display the top N results
result_text = customtkinter.CTkTextbox(app, width=300, height=100)
result_text.pack(padx=50, pady=50)

# Continually listening to tkinter event loop, such as button clikcs, mouse movements or keyboard inputs.
app.mainloop()
