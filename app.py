import streamlit as st
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import soundfile as sf
import tempfile

def extract_features(file):
    audio, sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

st.title("Voice Similarity Checker")

audio1 = st.file_uploader("Upload First Voice")
audio2 = st.file_uploader("Upload Second Voice")

if audio1 and audio2:

    t1 = tempfile.NamedTemporaryFile(delete=False)
    t1.write(audio1.read())

    t2 = tempfile.NamedTemporaryFile(delete=False)
    t2.write(audio2.read())

    voice1 = extract_features(t1.name)
    voice2 = extract_features(t2.name)

    similarity = cosine_similarity([voice1], [voice2])
    score = similarity[0][0] * 100

    st.write("Similarity:", round(score,2), "%")

    if score >= 80:
        st.success("Same Speaker")
    else:
        st.error("Different Speaker")