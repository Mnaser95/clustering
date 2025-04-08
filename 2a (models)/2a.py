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
from scipy.spatial import KDTree
from mne.preprocessing import ICA
from scipy.stats import spearmanr
from scipy.stats import kendalltau
##################################### Functions
def load_data(ses, data_type):
    my_file = fr"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\PhD-S7\Dissertation\BCI-Dissertation\After_Internship\Frontiers\Data\full_2a_data\Data\{ses}.mat"
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
    picks = ["8", "9", "14", "15", "11", "12", "17", "18"]     # the channels to consider (refer to data description)
    epoch_length_samples = int((tmax_rest-tmin_rest) * raw.info['sfreq'])
    n_samples = len(raw)

    # Creating a one event for the Rest period
    event_times = np.arange(0, n_samples - epoch_length_samples, epoch_length_samples)
    events = np.column_stack((event_times, np.zeros_like(event_times, dtype=int), np.ones_like(event_times, dtype=int)))
    epochs = mne.Epochs(raw, events, event_id=1, tmin=tmin_rest, tmax=tmax_rest, baseline=None, preload=True, picks=picks)
    
    data_rest_1  = np.mean(epochs.get_data()[0][:4, :], axis=0) # the first 4 channels are in the left hemisphere
    data_rest_2  = np.mean(epochs.get_data()[0][4:, :], axis=0) # the last 4 channels are in the right hemisphere
    
    _, _, Zxx1 = stft(data_rest_1, 250, nperseg=250) # generating time-frequency map using STFT
    _, _, Zxx2 = stft(data_rest_2, 250, nperseg=250) # generating time-frequency map using STFT
    
    return np.abs(Zxx1), np.abs(Zxx2), np.abs(Zxx1)-np.abs(Zxx2)
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
    return (avgs_x_l, avgs_y_l,avgs_x_r, avgs_y_r)
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
        vmin=-4,   # Set minimum value for color scale
        vmax=4    # Set maximum value for color scale
    )
    plt.colorbar(label="Value")
    plt.title(fr"{ses} Left hemi - Right hemi")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.ylim(0, 20)  # Ensure y-axis matches the full range
    plt.savefig(fr"Rest_{ses}.png")
    plt.close()  # Close the figure to free memory

    return()
def process_mi_data(raw, mat_data):
    raw.filter(f_low_MI, f_high_MI, fir_design='firwin') # FIR filtration to keep a range of frequencies

    events = np.squeeze(mat_data['data'][0][3][0][0][2]) # only the first run of each session is taken (total number of trials is 48, only left and right hand considered so 24)
    event_indices = np.squeeze(mat_data['data'][0][3][0][0][1])
    mne_events = np.column_stack((event_indices, np.zeros_like(event_indices), events))
    
    event_id_MI = dict({'769': 1, '770': 2})
    epochs_MI = mne.Epochs(raw, mne_events, event_id_MI, tmin_MI, tmax_MI, proj=True,  baseline=None, preload=True)
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
def rest_mi_relationships(mi_1_f,mi_2_f,rest_f):
    clusters = {
        0: [0,2,4,6,8,10,12,14,16,1,3,5,9,11,13,15,17],
    }

    for key, indices in clusters.items():
        mi_1 = np.abs([mi_1_f[i] for i in indices])
        mi_2 = np.abs([mi_2_f[i] for i in indices])
        rest = np.abs([rest_f[i] for i in indices])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

        # Plot 1: Left MI vs Rest Diff (Swapped Axes)
        model = LinearRegression()
        model.fit(np.array(rest).reshape(-1, 1), np.array(mi_1).reshape(-1, 1))  # Swap inputs
        predicted = model.predict(np.array(rest).reshape(-1, 1))
        r2 = r2_score(mi_1, predicted)
        spearman_corr1, _ = spearmanr(rest, mi_1)

        axes[0].scatter(rest, mi_1, color='blue', label='Data points')
        axes[0].plot(rest, predicted, color='red')
        axes[0].set_xlabel('|Left Hemi PSD - Right Hemi PSD| during Rest')
        axes[0].set_ylabel(r'|Left Hemi PSD - Right Hemi PSD| during LH')
        axes[0].grid(True)
        axes[0].set_xlim(0, 0.5)  # Adjust x-axis limits (previous y-axis)
        axes[0].set_ylim(0, 0.015)  # Adjust y-axis limits (previous x-axis)

        # Add R² text to the left plot
        axes[0].text(0.05, 0.012, f'$R^2$ = {r2:.3f}', fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))

        # Plot 2: Right MI vs Rest Diff (Swapped Axes)
        model = LinearRegression()
        model.fit(np.array(rest).reshape(-1, 1), np.array(mi_2).reshape(-1, 1))  # Swap inputs
        predicted = model.predict(np.array(rest).reshape(-1, 1))
        r2 = r2_score(mi_2, predicted)
        spearman_corr2, _ = spearmanr(rest, mi_2)

        axes[1].scatter(rest, mi_2, color='blue', label='Data points')
        axes[1].plot(rest, predicted, color='red')
        axes[1].set_xlabel('|Left Hemi PSD - Right Hemi PSD| during Rest')
        axes[1].set_ylabel(r'|Left Hemi PSD - Right Hemi PSD| during RH}')
        axes[1].grid(True)
        axes[1].set_xlim(0, 0.5)  # Adjust x-axis limits (previous y-axis)
        axes[1].set_ylim(0, 0.015)  # Adjust y-axis limits (previous x-axis)

        # Add R² text to the right plot
        axes[1].text(0.05, 0.012, f'$R^2$ = {r2:.3f}', fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))

        # Adjust layout and save the combined plot
        plt.tight_layout()
        plt.savefig(fr"Combined_corr_{key}.png")
        plt.clf()

    return()
def multiple_regression(my_dependent_variables,my_independent_variables):
    model = LinearRegression()
    model.fit(my_independent_variables, my_dependent_variables)
    return model
def corrs():
    values_group1 = [data_rest_diff_tf_abs_avg_time_and_desired_freq_dic[k] for k in data_rest_diff_tf_abs_avg_time_and_desired_freq_dic if k in keys_group1]
    values_group2 = [data_rest_diff_tf_abs_avg_time_and_desired_freq_dic[k] for k in data_rest_diff_tf_abs_avg_time_and_desired_freq_dic if k in keys_group2]
    f_stat, p_value = f_oneway(values_group1, values_group2)
    plt.figure(figsize=(8, 6))
    plt.boxplot([values_group1, values_group2], labels=["A", "B"])
    plt.ylabel("Rest diff")
    plt.title(fr"f:{f_stat},pval:{p_value}")
    plt.savefig("C_anova.png")

    values_group1 = [sp_mets_left[k] for k in sp_mets_left if k in keys_group1]
    values_group2 = [sp_mets_left[k] for k in sp_mets_left if k in keys_group2]
    f_stat, p_value = f_oneway(values_group1, values_group2)
    plt.figure(figsize=(8, 6))
    plt.boxplot([values_group1, values_group2], labels=["A", "B"])
    plt.ylabel("sp_mets_left")
    plt.title(fr"f:{f_stat},pval:{p_value}")
    plt.savefig("sp_mets_left.png")

    values_group1 = [sp_mets_right[k] for k in sp_mets_right if k in keys_group1]
    values_group2 = [sp_mets_right[k] for k in sp_mets_right if k in keys_group2]
    f_stat, p_value = f_oneway(values_group1, values_group2)
    plt.figure(figsize=(8, 6))
    plt.boxplot([values_group1, values_group2], labels=["A", "B"])
    plt.ylabel("sp_mets_right")
    plt.title(fr"f:{f_stat},pval:{p_value}")
    plt.savefig("sp_mets_right.png")

    return()
def tl():
    clusters =[list(keys_group1),list(keys_group2)]
    acc_return=[]
    for cluster in clusters:
        accuracies=[]
        all_keys= cluster
        for test_i in all_keys:
            training_keys= [key for key in all_keys if key != test_i]
            if len(training_keys)!=1:
                training_models=np.array([MI_models[a] for a in training_keys]).squeeze()
            else:
                training_models=np.array(MI_models[training_keys[0]])
            training_rest_features=np.array([data_rest_diff_tf_abs_dic[a] for a in training_keys])
            training_rest_features_tf=np.mean(training_rest_features[:,6:14,60:70],axis=2).reshape(training_rest_features.shape[0],-1)
            

            model=multiple_regression(training_models,training_rest_features_tf)
            www=np.mean(data_rest_diff_tf_abs_dic[test_i][6:14,60:70],axis=1)[np.newaxis, :]
            y_pred=model.predict(www)
            lda_constructed = LDA()
            lda_constructed.coef_ = np.array([y_pred[0,0:25]])
            lda_constructed.intercept_ = np.array([y_pred[0,-1]])
            lda_constructed.classes_ =  np.array([1,2])
            # 6:14 is the frequency, the other is time
            y_pred_y=lda_constructed.predict(data_MI_dic[test_i].squeeze())

            accuracies.append(accuracy_score(labels_MI_dic[test_i], y_pred_y))

            plt.plot(lda_constructed.coef_.ravel(),label="constructed")
            plt.plot(orig_classifiers[test_i].coef_.ravel(),label="original")
            plt.legend()    
            plt.savefig(fr"classifiers_{test_i}")
            plt.clf()

        acc_mean=np.mean(accuracies)
        acc_return.append(acc_mean)
    return(acc_return)
def get_sparsity(data_mi_stacked,labels_MI):
    unique_labels = np.unique(labels_MI)
    for label in unique_labels:
        # Get points corresponding to the current label
        label_points = data_mi_stacked[labels_MI == label]
        
        tree = KDTree(label_points)
        distances, _ = tree.query(data_mi_stacked, k=2)  # Find the nearest neighbor
        nearest_distances = distances[:, 1]  # Exclude self-distance
        sparsity_metrics[label] = np.mean(nearest_distances)
    return(sparsity_metrics[1],sparsity_metrics[2])
def plotting_both(data,y,classifier_constructed,classifier_original,testing_sub,orig_acc,cons_acc):
    x0=data[:,0];x1=data[:,1]
    plt.figure()
    plt.scatter(x0, x1, c=y, cmap='viridis', edgecolors='k')

    # Create grid to evaluate model
    xx, yy = np.meshgrid(np.linspace(-7, 7, 100),
                        np.linspace(-7, 7, 100))
    zz1 = classifier_original.predict(np.c_[xx.ravel(), yy.ravel()])
    #zz2 = classifier_constructed.predict(np.c_[xx.ravel(), yy.ravel()])
    zz1 = zz1.reshape(xx.shape)   
    #zz2 = zz2.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, zz1, alpha=0.5, cmap='viridis')
    #plt.contourf(xx, yy, zz2, alpha=0.5, cmap='viridis')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(fr"orig: {orig_acc}, cons:{cons_acc}")
    plt.savefig(fr"{testing_sub}_both")
    plt.clf()
    #plt.show()
    return()
def plotting_both_3(data,y,classifier_constructed,classifier_original,testing_sub,orig_acc,cons_acc):
   x0=data[:,0];x1=data[:,1];x2=data[:,2]
   # Plot data points
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   ax.scatter(x0, x1, x2, c=y, cmap='viridis', edgecolors='k')

   # Plot decision boundary
   xlim = ax.get_xlim()
   ylim = ax.get_ylim()
   zlim = ax.get_zlim()

   # Create grid to evaluate model
   xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                        np.linspace(ylim[0], ylim[1], 10))
   zz = (-classifier_constructed.intercept_[0] - classifier_constructed.coef_[0][0]*xx - classifier_constructed.coef_[0][1]*yy) / classifier_constructed.coef_[0][2]
   zz2 = (-classifier_original.intercept_[0] - classifier_original.coef_[0][0]*xx - classifier_original.coef_[0][1]*yy) / classifier_original.coef_[0][2]
    
   # Plot separating plane
   ax.plot_surface(xx, yy, zz, alpha=0.5, label='Constructed Decision Boundary')
   ax.plot_surface(xx, yy, zz2, alpha=0.5, label='Original Decision Boundary')

   plt.xlabel('Feature 1')
   plt.ylabel('Feature 2')
   ax.set_zlabel('Feature 3')
   plt.title(fr"{testing_sub}___orig: {orig_acc}, cons:{cons_acc}")
   ax.legend()
   plt.savefig("decision_boundary_{}.png".format(testing_sub))
   #plt.show()
   return()

##################################### Inputs
sfreq = 250 # sampling frequency
sess=[i for i in range(0,18)] # list of sessions (for all 9 subjects). Two sessions per subject so the total is 18.
f_low_rest=1   # low frequency
f_high_rest=15 # high frequency
f_low_MI=8   # low frequency
f_high_MI=12 # high frequency
tmin_rest = 1  # start of time for rest[s]
tmax_rest = 59 # end time for rest [s]
tmin_MI = 1
tmax_MI = 4
##################################### Data Prep
keys_group1 = {10,11,14,15,16} 
keys_group2 = {5,17} 


psd_pt_LH_left_hemi={}
psd_pt_LH_right_hemi={}
psd_pt_RH_left_hemi={}
psd_pt_RH_right_hemi={}
MI_models = {}
data_MI_dic= {}
left_mi_avg_over_time_and_freq={}
right_mi_avg_over_time_and_freq={}
data_rest_diff_tf_abs_avg_time_and_desired_freq_list=[]
data_rest_diff_tf_abs_avg_time_and_desired_freq_dic={}
data_rest_diff_tf_abs_dic={}
accuracies_psd={}
sp_mets_left={}
sp_mets_right={}
labels_MI_dic={}
sparsity_metrics={}
sp_mets_left={}
sp_mets_right={}
data_MI_original_dic={}
averages_left_l={}
averages_right_l={}
averages_left_r={}
averages_right_r={}
orig_classifiers={}
for ses in sess:
    ##########################################################
    # Process Rest Data
    ##########################################################
    rest_data, _= load_data(ses, 'rest')
    raw = create_mne_raw(rest_data)
    _,_, data_rest_diff_tf_abs = process_rest_data(raw) # number 1 is left hemisphere, 2 is right hemisphere
    plotting_rest_maps(data_rest_diff_tf_abs)
    
    data_rest_diff_tf_abs_avg_time=np.mean(np.squeeze(data_rest_diff_tf_abs),axis=1) 
    data_rest_diff_tf_abs_avg_time_and_desired_freq_list.append(np.mean(data_rest_diff_tf_abs_avg_time[7:13]))
    data_rest_diff_tf_abs_avg_time_and_desired_freq_dic[ses]=np.mean(data_rest_diff_tf_abs[7:13,:])
    data_rest_diff_tf_abs_dic[ses]=data_rest_diff_tf_abs
        

    ##########################################################
    # Process MI Data
    ##########################################################
    mi_data, mat_data = load_data(ses, 'mi')
    raw = create_mne_raw(mi_data)

    data_MI, labels_MI,data_MI_original,MI_tf = process_mi_data(raw, mat_data)
    data_MI_original_dic[ses]=data_MI_original
    data_MI_dic[ses]=data_MI     
    labels_MI_dic[ses]=labels_MI


    data_MI_tf_abs_avg_freq=np.mean(np.squeeze(MI_tf),axis=2) 
    data_MI_tf_abs_avg_freq_time=np.mean(np.squeeze(data_MI_tf_abs_avg_freq),axis=2) 

    data_MI_tf_abs_avg_freq_time_left_hemi=np.mean(data_MI_tf_abs_avg_freq_time[:, 0:4], axis=1) 
    data_MI_tf_abs_avg_freq_time_right_hemi=np.mean(data_MI_tf_abs_avg_freq_time[:, 4:8], axis=1) 
    

    left_d = [data_MI_tf_abs_avg_freq_time_left_hemi[i] - data_MI_tf_abs_avg_freq_time_right_hemi[i] for i in range(len(data_MI_tf_abs_avg_freq_time_left_hemi)) if labels_MI[i] == 1]
    right_d = [data_MI_tf_abs_avg_freq_time_left_hemi[i] - data_MI_tf_abs_avg_freq_time_right_hemi[i] for i in range(len(data_MI_tf_abs_avg_freq_time_left_hemi)) if labels_MI[i] == 2]
    left_mi_avg_over_time_and_freq[ses]=np.mean(left_d)
    right_mi_avg_over_time_and_freq[ses]=np.mean(right_d)

    data_mi_stacked_tf = np.vstack((data_MI_tf_abs_avg_freq_time_left_hemi, data_MI_tf_abs_avg_freq_time_right_hemi)).T  # Combine along the second axis

    model_parameters,accuracy_psd,orig_classifier=generate_training_MI_models(data_MI, labels_MI) # each session (only the first run) has an LDA classifier
    accuracies_psd[ses]=accuracy_psd
    MI_models[ses]=model_parameters
    orig_classifiers[ses]=orig_classifier
    
    data_mi_avg_left = np.mean(data_MI[:, 0:4, :], axis=1)  # Left hemisphere channels
    data_mi_avg_right = np.mean(data_MI[:, 4:8, :], axis=1)  # Right hemisphere channels
    data_mi_stacked = np.hstack((data_mi_avg_left, data_mi_avg_right))  # Combine along the second axis
    
    sp_left, sp_right=get_sparsity(data_mi_stacked,labels_MI)
    sp_mets_left[ses]=sp_left
    sp_mets_right[ses]=sp_right

    xs_l, ys_l, xs_r, ys_r=plotting_psds(data_mi_stacked_tf,labels_MI)

    averages_left_l[ses]=xs_l
    averages_right_l[ses]=ys_l
    averages_left_r[ses]=xs_r
    averages_right_r[ses]=ys_r


##########################################################
left_MI_psd_diff = np.array(list(averages_left_l.values())) - np.array(list(averages_right_l.values()))
right_MI_psd_diff = np.array(list(averages_left_r.values())) - np.array(list(averages_right_r.values()))


rest_mi_relationships(left_mi_avg_over_time_and_freq,right_mi_avg_over_time_and_freq,data_rest_diff_tf_abs_avg_time_and_desired_freq_dic)
corrs()
acc_cluster=tl()


# number 7 is wrong so don't consider it
stop=1
