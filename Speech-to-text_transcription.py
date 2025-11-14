pip install SpeechRecognition
pip install PyAudio
pip install pipwin
pipwin install pyaudio

import speech_recognition as sr

# Initialize recognizer
r = sr.Recognizer()

# Load the audio file
with sr.AudioFile("audio.wav") as source:
    audio = r.record(source) # read the entire audio file

try:
    # Recognize using Google's free Web Speech API
    text = r.recognize_google(audio)
    print("Transcription:")
    print(text)

except sr.UnknownValueError:
    print("Could not understand the audio")

except sr.RequestError:
    print("Error with the Speech Recognition service")