import mne
from scipy.signal import stft
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
import pandas as pd
from scipy.stats import pearsonr
from collections import Counter
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
warnings.filterwarnings("ignore")

N_SUBJECT = 109
BASELINE_EYE_CLOSED = [2]
IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST = [4, 8, 12]
SELECTED_CHANNELS = [8,9,1,2,15,16, 11,12,4,5,18,19]
num_chan_per_hemi=len(SELECTED_CHANNELS)//2
num_runs_sub=3
needed_subs=109
low_rest=1
high_rest=20
low_MI=8
high_MI=12
fs=126

#################################################################################################################################################### Rest
def raw_rest_processing(raw):
    raw.filter(l_freq=low_rest, h_freq=high_rest, picks="eeg", verbose='WARNING')
    events, _ = mne.events_from_annotations(raw)

    raw.rename_channels(lambda name: name.replace('.', '').strip().upper())
    df = pd.read_csv(fr"64montage.csv", header=None, names=["name", "x", "y", "z"])    
    ch_pos = {row['name']: [row['x'], row['y'], row['z']] for _, row in df.iterrows()}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage,on_missing="warn")
    valid_chs = [
    ch['ch_name'] for ch in raw.info['chs']
    if ch['loc'] is not None and not np.allclose(ch['loc'][:3], 0) and not np.isnan(ch['loc'][:3]).any()
    ]
    raw= raw.copy().pick_channels(valid_chs)

    raw = mne.preprocessing.compute_current_source_density(raw)
    #raw.plot_sensors(show_names=True)  # just to verify positions

    epoched = mne.Epochs(raw,events,event_id=dict(rest=1),tmin=1,tmax=59,proj=False,picks=SELECTED_CHANNELS,baseline=None,preload=True)

    return epoched
def rest_data_generation(epoched):
    X = (epoched.get_data() * 1e3).astype(np.float32)

    avg_left = X[:, :num_chan_per_hemi, :].mean(axis=1)  # (n_samples, time)
    avg_right = X[:, -num_chan_per_hemi:, :].mean(axis=1)

    # Apply STFT per sample and average across samples
    Zxx1_total = []
    Zxx2_total = []
    f, _, Z1 = stft(avg_left[0], fs, nperseg=fs)
    f, _, Z2 = stft(avg_right[0], fs, nperseg=fs)
    Zxx1_total.append(np.abs(Z1))
    Zxx2_total.append(np.abs(Z2))

    Zxx1_mean = np.mean(Zxx1_total, axis=0)
    Zxx2_mean = np.mean(Zxx2_total, axis=0)
    data = (Zxx1_mean - Zxx2_mean).squeeze()

    return data
def subject_select(mid,other):
    segment_size = 10
    n_segments = len(mid) // segment_size
    ratio = [mid[i]/other[i] for i in range(n_segments)]

    #"11": strong, positive (mainly yellow)
    #"10": strong, negative (mainly blue)
    #"0X": weak 

    votes=[]
    for ratio in ratio:
        if ratio > needed_ratio:
            votes.append("11") 
        elif ratio < -needed_ratio:
            votes.append("10")
        else:
            votes.append("0X")


    vote_counts = Counter(votes)
    majority_vote, num_majority_votes = vote_counts.most_common(1)[0]


    if majority_vote=="11":
        res="Pattern B"
    if majority_vote=="10":
        res="Pattern A"
    if majority_vote=="0X":
        res="Weak"
    confidence=num_majority_votes/n_segments
    return res, confidence
def rest_plotting(data,res_up,res_down,confidence_up,confidence_down,sub):
        # Define a custom colormap
        colors = [
            (0, 'blue'),
            (0.3, 'skyblue'),
            (0.5, 'black'),
            (0.7, 'lightyellow'),
            (1, 'yellow')
        ]
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        
        abs_greater=max(abs(np.min(data)),abs(np.max(data)))
        # Plot heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(
            data,
            aspect='auto',
            cmap=custom_cmap,
            interpolation='nearest',
            vmin=abs_greater,
            vmax=-abs_greater
        )
        plt.colorbar(label="Value")
        plt.title(f"Subject {sub+1} - Left hemi vs Right hemi")
        plt.text(40, 10, f"with up: {res_up}, confidence: {confidence_up}", fontsize=12,color="white")  
        plt.text(40, 5, f"with down: {res_down}, confidence: {confidence_down}", fontsize=12,color="white")  
        
        plt.xlabel("Timepoint")
        plt.ylabel("Frequency")
        plt.ylim(0, 20)
        plt.savefig(fr"rest_subject_{sub+1}.png")
        plt.close()

# Download/load data paths
physionet_paths = [mne.datasets.eegbci.load_data(subject_id,BASELINE_EYE_CLOSED,"/root/mne_data" ) for subject_id in range(1, needed_subs + 1) ]
physionet_paths = np.concatenate(physionet_paths)

# Read EDF files
parts = [mne.io.read_raw_edf(path,preload=True,stim_channel='auto',verbose='WARNING')for path in physionet_paths]

# Process each subject (every 1 file per subject in this case)
labels={}
patterns={}
confidences={}
confidences_multi={}
needed_ratio=1.2


for sub, raw in enumerate(parts):
    epoched=raw_rest_processing(raw)

    data=rest_data_generation(epoched)

    avg_mid_freq=data[8:11,:].mean(axis=0)
    avg_below_freq=data[2:8,:].mean(axis=0)
    avg_above_freq=data[11:15,:].mean(axis=0)
    res_down, confidence_down=subject_select(avg_mid_freq,avg_below_freq)
    res_up, confidence_up=subject_select(avg_mid_freq,avg_above_freq)
    if confidence_up > confidence_down:
        patterns[sub+1] = res_up
        confidences[sub+1] = confidence_up
    else:
        patterns[sub+1]  = res_down
        confidences[sub+1] = confidence_down

    #rest_plotting(data,res_up,res_down,confidence_up,confidence_down,sub)

for k in [k for k, v in patterns.items() if v == "Weak"]:
    del patterns[k]
    del confidences[k]

for k in [38,88,89,92,100,104]:
    if k in patterns:
        del patterns[k]
        del confidences[k]


subs_taken=list(patterns.keys())
subs_pattern_B = [k for k, v in patterns.items() if v == "Pattern B"]
confidences_multi = {k: (v if patterns[k] == "Pattern B" else -v) for k, v in confidences.items()}

stop=1
#################################################################################################################################################### MI
def raw_MI_processing(raw):
    raw.filter(l_freq=low_MI, h_freq=high_MI, picks="eeg", verbose='WARNING')
    
    raw.rename_channels(lambda name: name.replace('.', '').strip().upper())
    
    df = pd.read_csv(fr"64montage.csv", header=None, names=["name", "x", "y", "z"])    
    ch_pos = {row['name']: [row['x'], row['y'], row['z']] for _, row in df.iterrows()}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage,on_missing="warn")
    valid_chs = [
    ch['ch_name'] for ch in raw.info['chs']
    if ch['loc'] is not None and not np.allclose(ch['loc'][:3], 0) and not np.isnan(ch['loc'][:3]).any()
    ]
    raw= raw.copy().pick_channels(valid_chs)

    events, _ = mne.events_from_annotations(raw)

    epoched = mne.Epochs(raw, events, dict(left=2, right=3), tmin=1, tmax=4.1,proj=False, picks=SELECTED_CHANNELS, baseline=None, preload=True)
    return epoched
def MI_data_generation(epoched):
    X = (epoched.get_data() * 1e3).astype(np.float32)
    y = (epoched.events[:, 2] - 2).astype(np.int64)

    _, _, Zxx = stft(X, fs, nperseg=fs)
    MI_tf = np.abs(Zxx)
    X_tf = MI_tf.mean(axis=2).mean(axis=2)

    avg_left = X_tf[:, :num_chan_per_hemi].mean(axis=1)
    avg_right = X_tf[:, -num_chan_per_hemi:].mean(axis=1)
    X_avg = np.stack((avg_left, avg_right), axis=1)

    x_left = X_avg[:, 0]
    x_right = X_avg[:, 1]

    return x_left,x_right,y
def res_plotting(x_vals,y_vals,keys,subs_pattern_B):
    # Plot
    plt.clf()

    # Add x=0 and y=0 lines
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    # plt.xscale('symlog', linthresh=.00001)  
    # plt.yscale('symlog', linthresh=.00001)

    # Plot points and labels with appropriate color
    for x, y, label in zip(x_vals, y_vals, keys):
        color = 'black' if label in subs_pattern_B else 'red'
        plt.scatter(x, y, color=color)
        # plt.xscale('symlog', linthresh=1e-5)  # Adjust linthresh for better visibility
        # plt.yscale('symlog', linthresh=1e-5)
        plt.text(x, y, label, fontsize=9, ha='right', va='bottom', color=color)
    plt.savefig(fr"final")
def res_stats(x_vals,y_vals,confidences_multi):
    r_x, p_value_x = pearsonr(np.array(x_vals).flatten(),  np.array(list(confidences_multi.values())).flatten())
    r_y, p_value_y = pearsonr(np.array(y_vals).flatten(),  np.array(list(confidences_multi.values())).flatten())



    X = np.column_stack((x_vals, y_vals))  # shape (n_samples, 2)
    y = np.array(list(confidences_multi.values()))

    model = LinearRegression()
    model.fit(X, y)

    #print("R²:", model.score(X, y))  # overall correlation
    #print("Coefficients:", model.coef_)
    X = sm.add_constant(X)  # adds intercept
    model = sm.OLS(y, X).fit()
    print(model.summary())
    stop=1
def run_plot(x_left,x_right):

    mask_blue = y == 0
    mask_yellow = y == 1

    ax = axs[local_run_index]

    # Add x = y line
    min_val = min(x_left.min(), x_right.min())
    max_val = max(x_left.max(), x_right.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)

    # Add orthogonal dashed line
    mid_val = (min_val + max_val) / 2
    ax.plot([min_val, max_val], [2 * mid_val - min_val, 2 * mid_val - max_val], 'r--', linewidth=1)

    ax.scatter(x_left[mask_blue], x_right[mask_blue], color='blue', label='LH_MI')
    ax.scatter(x_left[mask_yellow], x_right[mask_yellow], color='yellow', label='RH-MI')

    # Compute and plot centers
    center_blue = (x_left[mask_blue].mean(), x_right[mask_blue].mean())
    center_yellow = (x_left[mask_yellow].mean(), x_right[mask_yellow].mean())
    dx = center_yellow[0] - center_blue[0]
    dy = center_yellow[1] - center_blue[1]

    delta_x_list.append(dx)
    delta_y_list.append(dy)

    ax.scatter(*center_blue, color='darkblue', edgecolor='white', s=100, label='Center LH_MI', marker='s', zorder=5)
    ax.scatter(*center_yellow, color='orange', edgecolor='black', s=100, label='Center RH_MI', marker='s', zorder=5)

    ax.set_title(f"Run {global_run_idx}")
    ax.set_xlabel("X")
    if local_run_index == 0:
        ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend()

#Load PhysioNet paths
physionet_paths = [ mne.datasets.eegbci.load_data(id,IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST,"/root/mne_data",) for id in range(1, needed_subs + 1)  ]
physionet_paths = np.concatenate(physionet_paths)

# Read EDF files
raws = [mne.io.read_raw_edf(path,preload=True,stim_channel='auto',verbose='WARNING',) for path in physionet_paths]

# Process runs in groups of 3 (each subject)
my_dic_x={}
my_dic_y={}
my_dic_acc={}
for beg_global_run_idx in range(0, len(raws), num_runs_sub):
    #fig, axs = plt.subplots(1, num_runs_sub, figsize=(15, 5), sharex=True, sharey=True)
    delta_x_list = []
    delta_y_list = []
    for local_run_index in range(num_runs_sub): 
        global_run_idx = beg_global_run_idx + local_run_index
        if global_run_idx >= len(raws):
            break  
        raw = raws[global_run_idx]
        epoched=raw_MI_processing(raw)
        x_left,x_right,y=MI_data_generation(epoched)


        # Compute and plot centers
        mask_blue = y == 0
        mask_yellow = y == 1
        center_blue = (x_left[mask_blue].mean(), x_right[mask_blue].mean())
        center_yellow = (x_left[mask_yellow].mean(), x_right[mask_yellow].mean())
        dx = center_yellow[0] - center_blue[0]
        dy = center_yellow[1] - center_blue[1]
            
        delta_x_list.append(dx)
        delta_y_list.append(dy)

        #run_plot(x_left,x_right)

    sub = beg_global_run_idx // num_runs_sub + 1
    if sub in subs_taken:
        my_dic_x[sub]=np.mean(delta_x_list)
        my_dic_y[sub]=np.mean(delta_y_list)

        # fig.suptitle(f"2D Plots - Subject {sub} | Avg ΔX: {np.mean(delta_x_list):.8f}, Avg ΔY: {np.mean(delta_y_list):.8f}")
        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.savefig(fr"MI_subject_{sub}.png")
        # plt.close()



x_vals = [my_dic_x[k] for k in my_dic_x]
y_vals = [my_dic_y[k] for k in my_dic_y]  
keys = list(my_dic_y.keys())

res_plotting(x_vals,y_vals,keys,subs_pattern_B)
res_stats(x_vals,y_vals,confidences_multi)



stop=1










