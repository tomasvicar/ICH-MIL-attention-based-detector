class Config:
    
    tmp_save_dir='../models_python'
    
    # train_num_workers=6
    # test_num_workers=3
    
    train_num_workers=0
    test_num_workers=0
    
    
    data_path='../raw2'
    data_table_path = '../VFN_Annotations.csv'
    
    
    model_name='model'
    
    train_batch_size = 2
    test_batch_size = 2
    
    max_epochs = 12
    step_size=5
    gamma=0.1
    init_lr=0.001
    
    crop_size_train = [416,416,25]
    crop_size_valid = [416,416,25]
    
    
    SPLIT_RATIO = [7,3]
    
    