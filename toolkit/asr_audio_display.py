import librosa
from IPython.display import Audio
from IPython.display import display

from toolkit.asrhelper import AudioSample


def display_audio(audio_file):
    audio, sample_rate = librosa.load(audio_file)
    display(Audio(audio_file, rate=sample_rate))


def display_audio_sample(sample: AudioSample):
    audio, sample_rate = librosa.load(sample.path)
    display(Audio(sample.path, rate=sample_rate))


def play_bip():
    display(Audio('toolkit/sound_attention.wav', autoplay=True))
