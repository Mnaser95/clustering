
def run_experiments(subs_considered,full_res):

    #!/usr/bin/env python
    # -*- coding: utf-8 -*-

    __author__ = "Karel Roots"

    import os
    import sys

    import numpy as np
    from EEGModels import get_models
    from data_loader import load_data
    from experiment_data import Experiment
    from tensorflow.keras import backend as K
    from tensorflow.keras.utils import to_categorical
    from training_testing import run_experiment
    import random
    import tensorflow as tf


    def set_seed(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


    set_seed(42)

    nr_of_epochs = 25
    nb_classes = 2
    trial_type = 2 # 2: imagined
    use_cpu = True 
    base_folder=fr'C:\\Users\\mohd9\\OneDrive - Kennesaw State University\\Desktop\\EEGMotorImagery-master\\data\\'

    # Loading data from files
    X, y = load_data(FNAMES=subs_considered, trial_type=trial_type, chunk_data=True, 
                    chunks=8, base_folder=base_folder, sample_rate=160,
                    samples=640,cpu_format=use_cpu,
                    preprocessing=True, hp_freq=0.5, bp_low=2, bp_high=60, notch=True,
                    hp_filter=False, bp_filter=True, artifact_removal=True)

    # Data formatting
    if use_cpu:
        print("Using CPU")
        K.set_image_data_format('channels_last')
        samples = X.shape[1]
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    else:
        print("Using GPU")
        K.set_image_data_format('channels_first')
        samples = X.shape[2]
        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    y = to_categorical(y, nb_classes)

    my_experiment = Experiment(trial_type, 'Num1', get_models(trial_type, nb_classes, samples, use_cpu), 
                            nr_of_epochs,
                            0.125, 0.2)

    temp_res=run_experiment(X, y, my_experiment)
    return temp_res