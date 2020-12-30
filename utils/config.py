class Config:
    base_dir = './'
    num_workers = 2
    patience = 25

    # FNN
    checkpoint_output = 'FNN_checkpoint.pth'
    feature_dir = 'output/features'
    checkpoint_path_FNN = 'FNN_checkpoint.pth'
    
    # CNN_1D
    batch_size_1D = 8
    checkpoint_path_1D = '1D_CNN_checkpoint.pth'

    # CNN_2D
    batch_size_2D = 32
    checkpoint_path_2D = '2D_CNN_checkpoint.pth'

    

