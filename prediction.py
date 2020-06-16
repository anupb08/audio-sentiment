import numpy as np
import os
import sys
import librosa
from keras.models import load_model
import pickle

#input_file = sys.argv[1]

emotions = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
f2 = open('model/feature8k.pkl','rb')
model = load_model('model/mlp_relu_adadelta_model.h5') 

def pred_emotion(input_file):
    test_file_path = input_file #"input/test.wav"#sys.argv[2]
    X,sr = librosa.load(test_file_path, sr = None)
    stft = np.abs(librosa.stft(X))
    
    ############# EXTRACTING AUDIO FEATURES #############
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40),axis=1)
    
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T,axis=0)
    
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr,fmin=0.5*sr* 2**(-6)).T,axis=0)
    
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sr*2).T,axis=0)
    
    features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    
    feature_all = pickle.load(f2)
    
    feature_all = np.vstack([feature_all,features])
    
    x_chunk = np.array(features)
    x_chunk = x_chunk.reshape(1,np.shape(x_chunk)[0])
    y_chunk_model1 = model.predict(x_chunk)
    index = np.argmax(y_chunk_model1)
    print('Emotion:',emotions[index])
    return emotions[index]
#pred_emotion('uploads/3m3f.st_ns5db.pcm128000.wav')
