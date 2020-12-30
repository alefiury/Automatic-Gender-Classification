import utils.process_data as process_data
from utils.config import Config

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import os
import librosa
import numpy as np
import tqdm
from pathlib import Path

import tensorflow as tf

import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic

import torch
from fairseq.models.wav2vec import Wav2VecModel

def extract_fundamental_frenq_yappt(sample, sample_rate):
    signal = basic.SignalObj(sample, sample_rate)
    pitch = pYAAPT.yaapt(signal)

    # YAAPT pitches 
    pitchY = pYAAPT.yaapt(signal, frame_length=40, tda_frame_length=40, f0_min=75, f0_max=600)

    return pitchY.samp_interp

def extract_wav_2_vec(sample, sample_rate, model):
    audio_input = torch.from_numpy(sample).unsqueeze(0)
    z = model.feature_extractor(audio_input)
    c = model.feature_aggregator(z).detach().numpy()
    
    return c

def extract_features(line, model, data_class):    
    sample_duration = 1

    file = line['file']
    label = int(line['label'])
    folder = os.path.join(Config.base_dir, data_class, 'male') if label == 1 else os.path.join(Config.base_dir, data_class, 'female')
    
    # Sets the name to be the path to where the file is in my computer
    filepath = os.path.join(folder, str(file))

    # Loads the audio file
    X, sample_rate = librosa.load(filepath, res_type='kaiser_fast') 

    sample = X

    if librosa.get_duration(sample, sample_rate) < sample_duration:
        sample = np.pad( sample, (0, (sample_rate*sample_duration) - len(sample)) )
    else:
        sample = sample[:sample_duration*sample_rate]

    mfccs = np.mean(librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=40).T, axis=0)

    fund_freq = extract_fundamental_frenq_yappt(sample, sample_rate)

    vec = np.mean(extract_wav_2_vec(sample, sample_rate, model).T, axis=0)
    
    features = np.concatenate((mfccs, vec, fund_freq), axis=None)
    
    return features, label
 
def load_features(path):
    """Load the features from a numpy array"""
    features_label = np.load(path, allow_pickle=True)

    return features_label

def concatenate_features(features_label):
    """Concatenate the features that were chosen earlier and loaded"""

    features = []

    for i in range(0, len(features_label)):
        features.append(features_label[i][0])

    return features

def concatenate_label(features_label):
    labels = []

    for i in range(0, len(features_label)):
        labels.append(features_label[i][1])

    np.unique(labels, return_counts=True)

    return labels

def setting_X_y(features, labels):
    X = np.array(features)

    y = np.array(labels)

    return X, y

def process_features_labels(features_label):

    features = concatenate_features(features_label)

    labels = concatenate_label(features_label)

    X, y = setting_X_y(features, labels)

    lb = LabelEncoder()

    y = tf.keras.utils.to_categorical(lb.fit_transform(y))

    ss = StandardScaler()
    X = ss.fit_transform(X)

    return X, y

def process_features(saved_features):
    features_label = saved_features
    features = concatenate_features(features_label)
    labels = concatenate_label(features_label)

    X, y = setting_X_y(features, labels)

    lb = LabelEncoder()

    y = tf.keras.utils.to_categorical(lb.fit_transform(y))

    X_train, X_test, X_val, y_train, y_test, y_val = process_data.divide_train_val_test(X, y)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_val = ss.transform(X_val)
    X_test = ss.transform(X_test)

    return X_train, X_test, X_val, y_train, y_test, y_val

def run_extraction(extraction_func, wav2vec_model_path, data_class):
    model = None
    if wav2vec_model_path is not None:
        cp = torch.load(wav2vec_model_path)
        model = Wav2VecModel.build_model(cp['args'], task=None)
        model.load_state_dict(cp['model'])

    df = process_data.concatenate_dfs(data_class)

    features_label = []

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        features_label.append(extraction_func(row, model, data_class))

    return features_label