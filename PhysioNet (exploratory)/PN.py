import mne
from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
import pandas as pd
from scipy.stats import pearsonr
from collections import Counter
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.decoding import CSP
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from mne.decoding import CSP  # Assuming you're using MNE's CSP
from sklearn.svm import SVC
warnings.filterwarnings("ignore")

baseline_session_id = [2]
MI_sessions = [4, 8, 12]
selected_channels = [8,9,1,15,16, 11,12,4,18,19]; num_chan_per_hemi=len(selected_channels)//2
needed_subs=109
num_runs_sub=3
take_nonA_B_C=False
only_A_B=True
plot_rest_flag=True
plot_MI_flag=True
fs=126
low_rest=1
high_rest=20
low_MI=8
high_MI=12

#################################################################################################################################################### Rest
def raw_rest_processing(raw):
    raw.filter(l_freq=low_rest, h_freq=high_rest, picks="eeg", verbose='WARNING')
    events, _ = mne.events_from_annotations(raw)
    raw.rename_channels(lambda name: name.replace('.', '').strip().upper())
    df = pd.read_csv(fr"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\PhD-S7\Dissertation\BCI-Dissertation\After_Internship\Frontiers\All_sessions\PhysioNet\64montage.csv", header=None, names=["name", "x", "y", "z"])    
    ch_pos = {row['name']: [row['x'], row['y'], row['z']] for _, row in df.iterrows()}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage,on_missing="warn")
    valid_chs = [ch['ch_name'] for ch in raw.info['chs'] if ch['loc'] is not None and not np.allclose(ch['loc'][:3], 0) and not np.isnan(ch['loc'][:3]).any()]
    raw= raw.copy().pick_channels(valid_chs)
    raw = mne.preprocessing.compute_current_source_density(raw)
    #raw.plot_sensors(show_names=True)  # just to verify positions
    epoched = mne.Epochs(raw,events,event_id=dict(rest=1),tmin=1,tmax=59,proj=False,picks=selected_channels,baseline=None,preload=True)
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
def subject_select():
    data_mid=data[8:11,:].mean(axis=0)
    data_below=data[2:8,:].mean(axis=0)
    data_above=data[11:15,:].mean(axis=0)

    segment_size = 10
    n_segments = len(data_mid) // segment_size
    ###################################################################################### below
    ratio = [data_mid[i]/data_below[i] for i in range(n_segments)]

    votes=[]
    for r in ratio:
        if r > needed_ratio:
            votes.append("11") 
        elif r < -needed_ratio:
            votes.append("10")
        else:
            if r <needed_ratio_weak and r >-needed_ratio_weak:
                votes.append("01")
            else:
                votes.append("00")

    vote_counts = Counter(votes)
    majority_vote, num_majority_votes = vote_counts.most_common(1)[0]

    if majority_vote=="11" and num_majority_votes>=7:
        res_below="Pattern B"
    elif majority_vote=="10" and num_majority_votes>=7:
        res_below="Pattern A"
    elif majority_vote=="01" and num_majority_votes>=7:
        res_below="Pattern C (weak)"
    else:
        res_below="ignore"
    confidence_below=num_majority_votes/n_segments

    ###################################################################################### above
    ratio = [data_mid[i]/data_above[i] for i in range(n_segments)]
    votes=[]
    for r in ratio:
        if r > needed_ratio:
            votes.append("11") 
        elif r < -needed_ratio:
            votes.append("10")
        else:
            if r <needed_ratio_weak and r >-needed_ratio_weak:
                votes.append("01")
            else:
                votes.append("00")

    vote_counts = Counter(votes)
    majority_vote, num_majority_votes = vote_counts.most_common(1)[0]

    if majority_vote=="11" and num_majority_votes>=7:
        res_above="Pattern B"
    elif majority_vote=="10" and num_majority_votes>=7:
        res_above="Pattern A"
    elif majority_vote=="01" and num_majority_votes>=7:
        res_above="Pattern C (weak)"
    else:
        res_above="ignore"
    confidence_above=num_majority_votes/n_segments

    #########################################################################
    if confidence_above > confidence_below:
        patterns[sub+1] = res_above
        confidences[sub+1] = confidence_above            
    else:
        patterns[sub+1] = res_below
        confidences[sub+1] = confidence_below     

    return 
def rest_plotting():
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
        plt.xlabel("Timepoint")
        plt.ylabel("Frequency")
        plt.ylim(0, 20)
        plt.savefig(fr"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\PhD-S7\Dissertation\BCI-Dissertation\After_Internship\Frontiers\All_sessions\PhysioNet\rest_subject_{sub+1}.png")
        plt.close()
def finalize_results():
    if only_A_B:
        for k in [k for k, v in patterns.items() if v == "ignore" or v =="Pattern C (weak)"]:
            del patterns[k]
            del confidences[k]
    else:
        if take_nonA_B_C:
            for i,pat in patterns.items():
                if pat=="ignore":
                    patterns[i]="Pattern C (weak)"
        else:
            for k in [k for k, v in patterns.items() if v == "ignore"]:
                del patterns[k]
                del confidences[k]


    subs_taken=list(patterns.keys())
    subs_pattern_B = [k for k, v in patterns.items() if v == "Pattern B"]
    subs_pattern_A = [k for k, v in patterns.items() if v == "Pattern A"]

    for element in patterns.values():
        if element=="Pattern A":
            label_stat.append(0) 
        if element=="Pattern B":
            label_stat.append(1) 
        if element=="Pattern C (weak)":
            label_stat.append(.5) 
    return subs_taken, subs_pattern_B, subs_pattern_A


physionet_paths = [mne.datasets.eegbci.load_data(subject_id,baseline_session_id,"/root/mne_data" ) for subject_id in range(1, needed_subs + 1) ]; physionet_paths = np.concatenate(physionet_paths)

parts = [mne.io.read_raw_edf(path,preload=True,stim_channel='auto',verbose='WARNING')for path in physionet_paths]

needed_ratio=1.35
needed_ratio_weak=1.35
patterns={}
confidences={}
confidences_multi={}
label_stat=[]

for sub, raw in enumerate(parts):
    epoched=raw_rest_processing(raw)

    data=rest_data_generation(epoched)

    subject_select()

    rest_plotting() if plot_rest_flag==True else None

subs_taken, subs_pattern_B, subs_pattern_A=finalize_results()
#final results needed for later on: patterns, confidences, label_stat, subs_taken, subs_pattern_B, subs_pattern_A

#################################################################################################################################################### MI
def raw_MI_processing(raw):
    raw.filter(l_freq=low_MI, h_freq=high_MI, picks="eeg", verbose='WARNING')
    
    raw.rename_channels(lambda name: name.replace('.', '').strip().upper())
    
    df = pd.read_csv(fr"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\PhD-S7\Dissertation\BCI-Dissertation\After_Internship\Frontiers\All_sessions\PhysioNet\64montage.csv", header=None, names=["name", "x", "y", "z"])    
    ch_pos = {row['name']: [row['x'], row['y'], row['z']] for _, row in df.iterrows()}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage,on_missing="warn")
    valid_chs = [
    ch['ch_name'] for ch in raw.info['chs']
    if ch['loc'] is not None and not np.allclose(ch['loc'][:3], 0) and not np.isnan(ch['loc'][:3]).any()
    ]
    raw= raw.copy().pick_channels(valid_chs)

    events, _ = mne.events_from_annotations(raw)

    epoched = mne.Epochs(raw, events, dict(left=2, right=3), tmin=1, tmax=4.1,proj=False, picks=selected_channels, baseline=None, preload=True)
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

def res_plotting(x_vals,y_vals,keys,subs_pattern_B,subs_pattern_A):
    # Plot
    plt.clf()

    # Add x=0 and y=0 lines
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    # plt.xscale('symlog', linthresh=.00001)  
    # plt.yscale('symlog', linthresh=.00001)

    # Plot points and labels with appropriate color
    for x, y, label in zip(x_vals, y_vals, keys):
        color = 'black' if label in subs_pattern_B else 'red' if label in subs_pattern_A else "blue"
        plt.scatter(x, y, color=color)
        # plt.xscale('symlog', linthresh=1e-5)  # Adjust linthresh for better visibility
        # plt.yscale('symlog', linthresh=1e-5)
        plt.text(x, y, label, fontsize=9, ha='right', va='bottom', color=color)
    plt.savefig(fr"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\PhD-S7\Dissertation\BCI-Dissertation\After_Internship\Frontiers\All_sessions\PhysioNet\final")
def res_stats(x_vals,y_vals,confidences_multi):
    r_x, p_value_x = pearsonr(np.array(x_vals).flatten(),  np.array(confidences_multi).flatten())
    r_y, p_value_y = pearsonr(np.array(y_vals).flatten(),  np.array(confidences_multi).flatten())

    X = np.column_stack((x_vals, y_vals))  # shape (n_samples, 2)
    y = np.array(confidences_multi).flatten()

    model = LinearRegression()
    model.fit(X, y)

    #print("R²:", model.score(X, y))  # overall correlation
    #print("Coefficients:", model.coef_)
    X = sm.add_constant(X)  # adds intercept
    model = sm.OLS(y, X).fit()
    print(model.summary())
    stop=1
def run_plot():

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

    delta_x_run.append(dx)
    delta_y_run.append(dy)

    ax.scatter(*center_blue, color='darkblue', edgecolor='white', s=100, label='Center LH_MI', marker='s', zorder=5)
    ax.scatter(*center_yellow, color='orange', edgecolor='black', s=100, label='Center RH_MI', marker='s', zorder=5)

    ax.set_title(f"Run {global_run_idx}")
    ax.set_xlabel("X")
    if local_run_index == 0:
        ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend()
def find_deltas():
    mask_blue = y == 0
    mask_yellow = y == 1
    center_blue = (x_left[mask_blue].mean(), x_right[mask_blue].mean())
    center_yellow = (x_left[mask_yellow].mean(), x_right[mask_yellow].mean())
    dx = center_yellow[0] - center_blue[0]
    dy = center_yellow[1] - center_blue[1]
        
    delta_x_run.append(dx)
    delta_y_run.append(dy)
def sub_plot():
    fig.suptitle(f"2D Plots - Subject {sub} | Avg ΔX: {np.mean(delta_x_run):.8f}, Avg ΔY: {np.mean(delta_y_run):.8f}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fr"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\PhD-S7\Dissertation\BCI-Dissertation\After_Internship\Frontiers\All_sessions\PhysioNet\MI_subject_{sub}.png")
    plt.close()
physionet_paths = [ mne.datasets.eegbci.load_data(id,MI_sessions,"/root/mne_data",) for id in range(1, needed_subs + 1)  ]; physionet_paths = np.concatenate(physionet_paths)

raws = [mne.io.read_raw_edf(path,preload=True,stim_channel='auto',verbose='WARNING',) for path in physionet_paths]

delta_x_sub={}
delta_y_sub={}
feature_dic={}
for beg_global_run_idx in range(0, len(raws), num_runs_sub):
    sub = beg_global_run_idx // num_runs_sub + 1
    if sub in subs_taken:    

        fig, axs = plt.subplots(1, num_runs_sub, figsize=(15, 5), sharex=True, sharey=True)

        delta_x_run = []
        delta_y_run = []
        for local_run_index in range(num_runs_sub): 
            global_run_idx = beg_global_run_idx + local_run_index
            raw = raws[global_run_idx]
            epoched=raw_MI_processing(raw)
            x_left,x_right,y=MI_data_generation(epoched)
            run_plot()
            find_deltas()

        delta_x_sub[sub]=np.mean(delta_x_run)
        delta_y_sub[sub]=np.mean(delta_y_run)
        sub_plot() if plot_MI_flag==True else None

x_vals = list(delta_x_sub.values())
y_vals = list(delta_y_sub.values())
res_plotting(x_vals,y_vals,subs_taken,subs_pattern_B,subs_pattern_A)
res_stats(x_vals,y_vals,label_stat)

stop=1