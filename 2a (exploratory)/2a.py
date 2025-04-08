####################################### Libraries
import numpy as np
import mne
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
##################################### Inputs
sfreq = 250 # sampling frequency
sess=[i for i in range(1,19)] # list of sessions (for all 9 subjects). Two sessions per subject so the total is 18.
f_low_rest=1   # low frequency
f_high_rest=15 # high frequency
f_low_MI=8   # low frequency
f_high_MI=12 # high frequency
tmin_rest = 1  # start of time for rest[s]
tmax_rest = 59 # end time for rest [s]
tmin_MI = 1
tmax_MI = 4
############################################################################## Rest
patterns={}
confidences={} # absolute
confidences_multi={} # both +ve and -ve
needed_ratio=1.2

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
def process_rest_data(raw):
    raw.filter(f_low_rest, f_high_rest, fir_design='firwin') # FIR filtration to keep a range of frequencies
    

    df = pd.read_csv(fr"25montage.csv", header=None, names=["name", "x", "y", "z"])
    ch_pos = {
        str(row['name']): np.array([row['x'], row['y'], row['z']])
        for _, row in df.iterrows()
    }

    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage, on_missing="warn")


    valid_chs = [
        ch['ch_name'] for ch in raw.info['chs']
        if ch['loc'] is not None
        and not np.allclose(ch['loc'][:3], 0)
        and not np.isnan(ch['loc'][:3]).any()
]

    raw = raw.copy().pick_channels(valid_chs)

    raw = mne.preprocessing.compute_current_source_density(raw)
    #raw.plot_sensors(show_names=True)  # just to verify positions
    

    
    picks = ["8", "9", "14", "15","2","3",   "11", "12", "17", "18", "5", "6"]     # the channels to consider (refer to data description)
    epoch_length_samples = int((tmax_rest-tmin_rest) * raw.info['sfreq'])
    n_samples = len(raw)

    # Creating a one event for the Rest period
    event_times = np.arange(0, n_samples - epoch_length_samples, epoch_length_samples)
    events = np.column_stack((event_times, np.zeros_like(event_times, dtype=int), np.ones_like(event_times, dtype=int)))
    epochs = mne.Epochs(raw, events, event_id=1, tmin=tmin_rest, tmax=tmax_rest, baseline=None, preload=True, picks=picks)
    
    data_rest_1  = np.mean(epochs.get_data()[0][:6, :], axis=0) # the first 4 channels are in the left hemisphere
    data_rest_2  = np.mean(epochs.get_data()[0][6:, :], axis=0) # the last 4 channels are in the right hemisphere
    
    f, _, Zxx1 = stft(data_rest_1, 250, nperseg=250) # generating time-frequency map using STFT
    f, _, Zxx2 = stft(data_rest_2, 250, nperseg=250) # generating time-frequency map using STFT
    
    return np.abs(Zxx1), np.abs(Zxx2), np.abs(Zxx1)-np.abs(Zxx2)
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
def plotting_rest_maps(data):
    
    # Define a custom colormap
    colors = [
        (0, 'blue'),   # Low values
        (0.3, 'skyblue'),  # Middle values (around 0)
        (0.5, 'black'),  # Middle values (around 0)
        (0.7, 'lightyellow'),  # Middle values (around 0)
        (1, 'yellow')       # High values
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(
        data,
        aspect='auto',
        cmap=custom_cmap,
        interpolation='nearest',
        vmin=-.004,   # Set minimum value for color scale
        vmax=.004    # Set maximum value for color scale
    )
    plt.colorbar(label="Value")
    plt.title(fr"{ses} Left hemi - Right hemi")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.ylim(0, 20)  # Ensure y-axis matches the full range
    plt.savefig(fr"Rest_{ses}.png")
    plt.close()  # Close the figure to free memory

    return()

for ses in sess:
    rest_data, _= load_data(ses, 'rest')
    raw = create_mne_raw(rest_data)
    _,_, data_rest_diff_tf_abs = process_rest_data(raw) # number 1 is left hemisphere, 2 is right hemisphere
    plotting_rest_maps(data_rest_diff_tf_abs)

    avg_mid_freq=np.mean(data_rest_diff_tf_abs[7:13,:],axis=0)
    avg_below_freq=np.mean(data_rest_diff_tf_abs[2:7,:],axis=0)
    avg_above_freq=np.mean(data_rest_diff_tf_abs[13:18,:],axis=0)

    res_down, confidence_down=subject_select(avg_mid_freq,avg_below_freq)
    res_up, confidence_up=subject_select(avg_mid_freq,avg_above_freq)

    if confidence_up >= confidence_down:
        patterns[ses] = res_up
        confidences[ses] = confidence_up
    else:
        patterns[ses]  = res_down
        confidences[ses] = confidence_down

for k in [k for k, v in patterns.items() if v == "Weak"]: #remove weak
    del patterns[k]
    del confidences[k]

subs_taken=list(patterns.keys())
subs_pattern_B = [k for k, v in patterns.items() if v == "Pattern B"]
confidences_multi = {k: (v if patterns[k] == "Pattern B" else -v) for k, v in confidences.items()}

stop=1


############################################################################## MI

def plotting_psds(data_received, labels_MI):
    unique_labels = np.unique(labels_MI)

    # Prepare scatter plot for individual points
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(data_received[:, 0], data_received[:, 1], c=labels_MI, cmap='viridis', s=100, label="Points")
    plt.colorbar(sc, label="Labels (MI)")

    # Add the x = y straight line
    x_min, x_max = plt.xlim()  # Get current x-axis limits
    plt.plot([x_min, x_max], [x_min, x_max], color='red', linestyle='--', label="x = y")  # Dashed red line

    # Add labels to each point
    for i in range(data_received.shape[0]):
        plt.text(
            data_received[i, 0],  # X-coordinate
            data_received[i, 1],  # Y-coordinate
            str(i),  # Text label (index)
            fontsize=8,  # Font size
            ha='center',  # Horizontal alignment
            va='center',  # Vertical alignment
            color='black'
        )

    # Compute and plot average locations as circles
    for label in unique_labels:
        # Get points corresponding to the current label
        label_points = data_received[labels_MI == label]
        
        # Compute the average location for the label
        avg_x, avg_y = np.mean(label_points, axis=0)
        
        if label==1:
            avgs_x_l=avg_x
            avgs_y_l=avg_y
        else:
            avgs_x_r=avg_x
            avgs_y_r=avg_y            


        if label == 1:
            # Plot a circle at the average location
            plt.scatter(avg_x, avg_y, color='blue', edgecolor='black', s=200, label=f"LH MI", alpha=0.6)
            psd_pt_LH_left_hemi[ses]=avg_x
            psd_pt_LH_right_hemi[ses]=avg_y
        else:
            plt.scatter(avg_x, avg_y, color='yellow', edgecolor='black', s=200, label=f"RH MI", alpha=0.6)
            psd_pt_RH_left_hemi[ses]=avg_x
            psd_pt_RH_right_hemi[ses]=avg_y
    # Add labels and legend
    plt.xlabel("Left hemisphere")
    plt.ylabel("Right hemisphere")
    plt.title(fr"(2a) {ses}")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(fr"MI_{ses}.png")
    plt.clf()

    dx = avgs_x_r - avgs_x_l
    dy = avgs_y_r - avgs_y_l

    return (dx,dy)
def process_mi_data(raw, mat_data):
    raw.filter(f_low_MI, f_high_MI, fir_design='firwin') # FIR filtration to keep a range of frequencies



    df = pd.read_csv(fr"25montage.csv", header=None, names=["name", "x", "y", "z"])
    ch_pos = {
        str(row['name']): np.array([row['x'], row['y'], row['z']])
        for _, row in df.iterrows()
    }

    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage, on_missing="warn")


    valid_chs = [
        ch['ch_name'] for ch in raw.info['chs']
        if ch['loc'] is not None
        and not np.allclose(ch['loc'][:3], 0)
        and not np.isnan(ch['loc'][:3]).any()
]

    raw = raw.copy().pick_channels(valid_chs)

    raw = mne.preprocessing.compute_current_source_density(raw)



    events = np.squeeze(mat_data['data'][0][3][0][0][2]) # only the first run of each session is taken (total number of trials is 48, only left and right hand considered so 24)
    event_indices = np.squeeze(mat_data['data'][0][3][0][0][1])
    mne_events = np.column_stack((event_indices, np.zeros_like(event_indices), events))
    picks = ["8", "9", "14", "15","2","3",   "11", "12", "17", "18", "5", "6"] 

    event_id_MI = dict({'769': 1, '770': 2})
    epochs_MI = mne.Epochs(raw, mne_events, event_id_MI, tmin_MI, tmax_MI, proj=True,  baseline=None, preload=True, picks=picks)
    labels_MI = epochs_MI.events[:, -1]
    data_MI_original = epochs_MI.get_data()

    # Downsampling (averaging) data to "needed_size" number of datapoints
    step=int((250*(tmax_MI-tmin_MI))/1)
    data_downsized=np.ndarray(shape=(len(data_MI_original),len(data_MI_original[1]),1))
    for i in range(len(data_MI_original)):
        for j in range(len(data_MI_original[1])):
            for k in range(1):
                data_downsized[i,j,k]=np.mean(data_MI_original[i,j,k*step:(k+1)*step])


    _, _, Zxx = stft(data_MI_original, 250, nperseg=250) # generating time-frequency map using STFT
    MI_tf=np.abs(Zxx)



    return (data_downsized,labels_MI,data_MI_original,MI_tf)
def generate_training_MI_models(data_MI,labels_MI):
    lda = LDA()  
    num_points= len(data_MI[:,0,0])             
    lda.fit(data_MI.reshape(num_points,-1), labels_MI) # the classifier is fit based on the flattened time-frequency features
    coeffs= lda.coef_
    intercept= lda.intercept_.reshape(1,-1)
    model_parameters = np.hstack((coeffs, intercept))
    predictions = lda.predict(data_MI.reshape(num_points, -1))
    accuracy_psd = accuracy_score(labels_MI, predictions)
    return (model_parameters,accuracy_psd,lda)
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
    print(r_x, p_value_x)
    print(r_y, p_value_y)

    X = np.column_stack((x_vals, y_vals))  # shape (n_samples, 2)
    y = np.array(list(confidences_multi.values()))

    model = LinearRegression()
    model.fit(X, y)

    X = sm.add_constant(X)  # adds intercept
    model = sm.OLS(y, X).fit()
    print(model.summary())
    stop=1

psd_pt_LH_left_hemi={}
psd_pt_LH_right_hemi={}
psd_pt_RH_left_hemi={}
psd_pt_RH_right_hemi={}
MI_models = {}
data_MI_dic= {}
delta_x_dic={}
delta_y_dic={}
my_dic_x={}
my_dic_y={}

for ses in sess:
    mi_data, mat_data = load_data(ses, 'mi')
    raw = create_mne_raw(mi_data)

    data_MI, labels_MI,data_MI_original,MI_tf = process_mi_data(raw, mat_data)
    data_MI_dic[ses]=data_MI     

    data_MI_tf_abs_avg_freq=np.mean(np.squeeze(MI_tf),axis=2) 
    data_MI_tf_abs_avg_freq_time=np.mean(np.squeeze(data_MI_tf_abs_avg_freq),axis=2) 

    data_MI_tf_abs_avg_freq_time_left_hemi=np.mean(data_MI_tf_abs_avg_freq_time[:, :6], axis=1) 
    data_MI_tf_abs_avg_freq_time_right_hemi=np.mean(data_MI_tf_abs_avg_freq_time[:, 6:], axis=1) 


    left_d = [data_MI_tf_abs_avg_freq_time_left_hemi[i] - data_MI_tf_abs_avg_freq_time_right_hemi[i] for i in range(len(data_MI_tf_abs_avg_freq_time_left_hemi)) if labels_MI[i] == 1]
    right_d = [data_MI_tf_abs_avg_freq_time_left_hemi[i] - data_MI_tf_abs_avg_freq_time_right_hemi[i] for i in range(len(data_MI_tf_abs_avg_freq_time_left_hemi)) if labels_MI[i] == 2]

    data_mi_stacked_tf = np.vstack((data_MI_tf_abs_avg_freq_time_left_hemi, data_MI_tf_abs_avg_freq_time_right_hemi)).T  # Combine along the second axis

    data_mi_avg_left = np.mean(data_MI[:, :6, :], axis=1)  # Left hemisphere channels
    data_mi_avg_right = np.mean(data_MI[:, 6:, :], axis=1)  # Right hemisphere channels
    data_mi_stacked = np.hstack((data_mi_avg_left, data_mi_avg_right))  # Combine along the second axis

    dx,dy=plotting_psds(data_mi_stacked_tf,labels_MI)
       
    delta_x_dic[ses]=dx
    delta_y_dic[ses]=dy

############################################################################## Both
for k in range(1,19):
    if k not in subs_taken:
        del delta_x_dic[k]
        del delta_y_dic[k]


x_vals = [delta_x_dic[k] for k in delta_x_dic]
y_vals = [delta_y_dic[k] for k in delta_y_dic]  
keys = list(delta_y_dic.keys())

res_plotting(x_vals,y_vals,keys,subs_pattern_B)
res_stats(x_vals,y_vals,confidences_multi)

# number 7 is wrong so don't consider it

# channel location
# why all is +ve
stop=1
