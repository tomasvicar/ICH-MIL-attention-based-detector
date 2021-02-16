import numpy as np
from resnet_2D_heatmap import Resnet_2D_heatmap

class Config:
    
    tmp_save_dir='../models_python'
    

    # train_num_workers=2
    # test_num_workers=2
    
    train_num_workers = 10
    test_num_workers = 10
    
    

    data_path='..'
    data_table_path = '../bBoxAnnotations_All_patients2.csv'
    
    
    model_name='model'
    
    train_batch_size = 32
    test_batch_size = 32
    
    
    # lr_steps = np.cumsum([50,20,10])
    lr_steps = np.cumsum([40,10,5])
    gamma = 0.1
    init_lr = 0.001
    max_epochs = lr_steps[-1]
    
    # net = Small_resnet3D_noheatmap
    net = Resnet_2D_heatmap

    
    SPLIT_RATIO = [8,2]
    
    ###### Augmentation parameters
    max_multiplier  = 0.1   # multiply augmentation 
    max_add  = 20          # Add augmentation
    scale_range = 0.1
    shears_range = 0.05
    tilt_range = 0
    translation_range = 50
    rotation_range = 30



        
    
    
