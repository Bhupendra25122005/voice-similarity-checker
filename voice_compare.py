import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def extract_features(file):
    audio, sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

voice1 = extract_features("audio1.wav")
voice2 = extract_features("audio2.wav")

similarity = cosine_similarity([voice1], [voice2])
score = similarity[0][0]

percentage = score * 100

print("Voice Similarity:", percentage, "%")

if percentage >= 80:
    print("Result: Same Speaker")
else:
    print("Result: Different Speaker")