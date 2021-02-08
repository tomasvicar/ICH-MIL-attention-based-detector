import numpy as np
class Config:
    
    tmp_save_dir='../models_python'
    
    # train_num_workers=6
    # test_num_workers=3
    
    train_num_workers=0
    test_num_workers=0
    
    
    # data_path='../raw2'
    # data_table_path = '../VFN_Annotations.csv'
    data_path='../raw_cq'
    data_table_path = '../CQ_Annotations.csv'
    
    model_name='model'
    
    train_batch_size = 1
    test_batch_size = 1
    
    
    lr_steps = np.cumsum([30,20,10])
    gamma = 0.1
    init_lr = 0.01
    max_epochs = lr_steps[-1]
    
    
    crop_size_train = [416,416,35]
    crop_size_valid = [416,416,35]
    
    # crop_size_train = None
    # crop_size_valid = None 
    
    
    SPLIT_RATIO = [7,3]
    
    