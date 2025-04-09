####################################### Libraries
import numpy as np
import mne
import sys
import scipy.io
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from scipy.signal import stft
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from collections import Counter
import pandas as pd
import statsmodels.api as sm
from preprocess import preprocess_data
from run_experiments import run_experiments
##################################### Inputs
sfreq = 250 # sampling frequency
#sess=[i for i in range(1,19)] # list of sessions (for all 9 subjects). Two sessions per subject so the total is 18.
#sess=[6,18,5,16,10] # list of sessions (for all 9 subjects). Two sessions per subject so the total is 18.
sess=[6,18,15,11,3]

f_low_MI=.5   # low frequency
f_high_MI=60 # high frequency
tmin_MI = 0
tmax_MI = 4

def preprocess_data(data, sample_rate=250, ac_freq=60, hp_freq=0.5, bp_low=2, bp_high=60, notch=False,
                    hp_filter=False, bp_filter=False, artifact_removal=False, normalize=False):
    if notch:
        data = notch_filter(data, ac_freq, sample_rate)
    if hp_filter:
        data = highpass_filter(data, hp_freq)
    if bp_filter:
        data = bandpass_filter(data, bp_low, bp_high, sample_rate)
    if normalize:
        data = normalize_data(data, 'mean_std')
    if artifact_removal:
        data = remove_artifacts(data)

    return data


def notch_filter(data, ac_freq, sample_rate):
    w0 = ac_freq / (sample_rate / 2)
    return signal.notch(data, w0)


def highpass_filter(data, hp_freq):
    return signal.butter_highpass(data, hp_freq)


def bandpass_filter(data, bp_low, bp_high, sample_rate):
    return signal.butter_bandpass(data, bp_low, bp_high, order=5, fs=sample_rate)


def normalize_data(data, strategy):
    return signal.normalize(data, strategy)


def remove_artifacts(data):
    cleaned = signal.artifact_removal(data.reshape((-1, 1)))[0]
    return np.squeeze(cleaned)

def load_data(ses, data_type):
    my_file = fr"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\PhD-S7\Dissertation\Data\2a2b data\full_2a_data\Data\{ses-1}.mat"
    mat_data = scipy.io.loadmat(my_file)
    if data_type == 'rest':
        my_data_eeg = np.squeeze(mat_data['data'][0][1][0][0][0][:, 0:22]) # the first 22 channels are EEG
        my_data_eog = np.squeeze(mat_data['data'][0][1][0][0][0][:, 22:25]) # the rest are EOG
    elif data_type == 'mi':
        my_data_eeg = np.squeeze(mat_data['data'][0][3][0][0][0][:, 0:22])
        my_data_eog = np.squeeze(mat_data['data'][0][3][0][0][0][:, 22:25])
    return np.hstack([my_data_eeg, my_data_eog]),mat_data
def create_mne_raw(data):
    numbers = list(range(1, 26))
    ch_names = [str(num) for num in numbers]
    ch_types = ['eeg'] * 22 + ['eog'] * 3
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data.T, info)
    return raw
def process_mi_data(raw, mat_data):
    raw.filter(f_low_MI, f_high_MI, fir_design='firwin') # FIR filtration to keep a range of frequencies
    raw.notch_filter(freqs=50, fir_design='firwin')

    events = np.squeeze(mat_data['data'][0][3][0][0][2]) # only the first run of each session is taken (total number of trials is 48, only left and right hand considered so 24)
    event_indices = np.squeeze(mat_data['data'][0][3][0][0][1])
    mne_events = np.column_stack((event_indices, np.zeros_like(event_indices), events))

    event_id_MI = dict({'769': 1, '770': 2})
    epochs_MI = mne.Epochs(raw, mne_events, event_id_MI, tmin_MI, tmax_MI, proj=True,  baseline=None, preload=True)
    labels_MI = epochs_MI.events[:, -1]
    data_MI_original = epochs_MI.get_data()

    for trial in range(data_MI_original.shape[0]):
        for ch in range(25):
            data_trial=data_MI_original[trial,ch,:]

            data_clean=preprocess_data(data_trial)
            data_MI_original[trial, ch, :] = data_clean

    return (labels_MI,data_MI_original)

all_ses_data=[]
all_ses_labels=[]

for ses in sess:
    mi_data, mat_data = load_data(ses, 'mi')
    raw = create_mne_raw(mi_data)

    labels_MI,data_MI_original = process_mi_data(raw, mat_data)
    all_ses_data.append(data_MI_original)     
    all_ses_labels.append(labels_MI)  

all_ses_data_arr=np.array(all_ses_data)    
all_ses_labels_arr=np.array(all_ses_labels)   

all_ses_data_arr_reshaped=all_ses_data_arr.reshape(-1,all_ses_data_arr.shape[2],all_ses_data_arr.shape[3])
data_ready=all_ses_data_arr_reshaped.swapaxes( 1, 2)
labels_ready=all_ses_labels_arr.reshape(-1)


split_data = []
split_labels = []

for i in range(len(data_ready)):
    split_data.append(data_ready[i,:500,:])
    split_data.append(data_ready[i,500:-1,:])
    split_labels.append(labels_ready[i])
    split_labels.append(labels_ready[i])

split_data_ready=np.stack(split_data)
split_labels_ready=np.stack(split_labels)

temp_res=run_experiments(split_data_ready,split_labels_ready)

stop=1