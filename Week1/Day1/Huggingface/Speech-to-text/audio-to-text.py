from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", device = 0)

result = pipe("SOUND.mp3", return_timestamps=True )
print(result)
