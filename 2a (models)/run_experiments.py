
def run_experiments(X,y):

    #!/usr/bin/env python
    # -*- coding: utf-8 -*-

    __author__ = "Karel Roots"

    import os
    import sys

    import numpy as np
    from EEGModels import get_models
    #from data_loader import load_data
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

    nr_of_epochs = 15
    nb_classes = 2
    trial_type = 2 # 2: imagined
    use_cpu = True 
  
  
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
    
    y = y - 1 
    y = to_categorical(y, nb_classes)

    my_experiment = Experiment(trial_type, 'Num1', get_models(trial_type, nb_classes, samples, use_cpu), 
                            nr_of_epochs,
                            0.125, 0.2)

    temp_res=run_experiment(X, y, my_experiment)
    return temp_res