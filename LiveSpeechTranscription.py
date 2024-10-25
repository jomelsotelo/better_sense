import subprocess
import threading
import queue
import soundfile as sf
import numpy as np
import pyaudio
from faster_whisper import WhisperModel  # Import faster-whisper

# Constants
WAVE_OUTPUT_FILENAME = "output.wav"
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024
RECORD_SECONDS = 3
SILENCE_THRESHOLD = 0.02  # Energy threshold for silence detection

# Initialize the faster-whisper model
model = WhisperModel("tiny", device="cpu", compute_type="int8")  # Load the faster-whisper model

# Queue to hold audio files for transcription
audio_queue = queue.Queue()

def is_silent(audio_data):
    """Check if the audio data is silent based on energy threshold."""
    energy = np.sum(np.square(audio_data.astype(np.float32)))  # Calculate energy
    return energy < SILENCE_THRESHOLD

def transcribe_audio():
    while True:
        if not audio_queue.empty():
            filename = audio_queue.get()  # Get the audio file from the queue
            try:
                audio, _ = sf.read(filename)  # Read the audio file
                
                # Check if the audio is silent
                if is_silent(audio):
                    continue  # Skip silent audio

                segments, _ = model.transcribe(audio, language="en")  # Transcribe using faster-whisper
                
                # Process the segments and print transcriptions
                for segment in segments:
                    print(f"Transcription: {segment.text.strip()}")

            except Exception as e:
                print(f"Error during transcription: {e}")

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True,
                    frames_per_buffer=CHUNK)

    print("Recording... Press Ctrl+C to stop.")
    while True:
        frames = []
        for _ in range(0, int(SAMPLE_RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # Save recorded data to a WAV file
        wf = sf.SoundFile(WAVE_OUTPUT_FILENAME, 'w', samplerate=SAMPLE_RATE,
                          channels=CHANNELS, format='WAV')
        wf.write(np.frombuffer(b''.join(frames), dtype=np.int16))
        wf.close()

        # Put the filename in the queue for transcription
        audio_queue.put(WAVE_OUTPUT_FILENAME)

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    # Start the transcription thread
    transcription_thread = threading.Thread(target=transcribe_audio, daemon=True)
    transcription_thread.start()

    # Start the recording function
    try:
        record_audio()
    except KeyboardInterrupt:
        print("\nRecording stopped.")
