from scipy.io import wavfile
from pesq import pesq
import librosa


rate, ref = wavfile.read("/Users/hrishikeshbingewar/Downloads/av_project/GRID/s31/audio/bbav5s.wav")
rate, deg = wavfile.read("/Users/hrishikeshbingewar/Downloads/av_project/enhanced_speech.wav")

print(pesq(rate, ref, deg, 'wb'))
print(pesq(rate, ref, deg, 'nb'))