import numpy as np

import matplotlib.pyplot as plt
# import matplotlib.style as ms
# ms.use('seaborn-muted')
%matplotlib inline

# IPython.display for audio output
import IPython.display

# librosa for audio
import librosa
# display module for visualization
import librosa.display



# path to audio file
# audio_path = "/data/music/batch2/Moana (Original Motion Picture Soundtrack|Deluxe Edition)/04 How Far I'll Go.wav"
audio_path = "/data/music/batch2/Moana (Original Motion Picture Soundtrack|Deluxe Edition)/04 How Far I'll Go.wav"

# load the song
# offset-- start reading after this time (in seconds)
# duration-- load up to this much audio (in seconds)
y, sr = librosa.load(audio_path, offset=49.0, duration=25.0)

# play the song
IPython.display.Audio(data=y, rate=sr)
