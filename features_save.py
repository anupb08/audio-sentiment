import pickle
import sys
import numpy as np
import librosa
import os
path_data = sys.argv[1]
directories = os.listdir(path_data)

feature_all=np.empty((0,193))
for d in directories:
    for file in os.listdir(os.path.join(path_data,d)):
        #X,sr = librosa.load('/home/flash/Documents/IUBBooks/MLSP/Prooject/RAVDESS/Data/01/03-01-01-01-01-01-01.wav',sr = None)
        X,sr = librosa.load(os.path.join(path_data,d,file),duration=3)
        stft = np.abs(librosa.stft(X))
        #print("stft shape: ", np.shape(stft))
        #s.append(np.shape(stft))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)
        #mfccs=mfccs.reshape(1,np.shape(mfccs)[0])
        #print("mfccs shape: ", np.shape(mfccs))
        #m.append(np.shape(mfccs))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
        #chroma = chroma.reshape(1,np.shape(chroma)[0])
        #print("chroma shape: " , np.shape(chroma))
        #c.append(np.shape(chroma))
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T,axis=0)
       # mel = mel.reshape(1,np.shape(mel)[0])
        #print("mel shape:" , np.shape(mel))
        #me.append(np.shape(mel))
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
        #contrast = contrast.reshape(1,np.shape(contrast)[0])
        #print("contrast shape:",np.shape(contrast))
        #co.append(np.shape(contrast))
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sr).T,axis=0)
        #tonnetz = tonnetz.reshape(1,np.shape(tonnetz)[0])

        #print("tonnetz shape:",np.shape(tonnetz))
        #to.append(np.shape(tonnetz))
        features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        #features = features.reshape(np.shape(features)[0],1)
        feature_all = np.vstack([feature_all,features])
        #labels.append(d)
print(features.shape)
with open('model/feature8k.pkl', 'wb') as pickle_file:
    pickle.dump(feature_all, pickle_file)

