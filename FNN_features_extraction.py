from utils.process_data import concatenate_dfs
from utils.features_extraction import run_extraction, extract_features
import argparse

import numpy as np

import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_save', type=str, help='Path to save ndarray file') 
    parser.add_argument('-m', '--wav2vec_model_path', type=str, default='./utils/wav2vec_large.pt', help='Path to Wav2Vec model')  
    parser.add_argument('-t', '--features_train', help='Command to get features to train', action="store_true") 
    parser.add_argument('-e', '--features_eval', help='Command to get features to evaluate', action="store_true") 
    args = parser.parse_args()

    if args.features_train:
        features_label_train = run_extraction(extraction_func=extract_features, wav2vec_model_path=args.wav2vec_model_path, data_class='train')
        features_label_val = run_extraction(extraction_func=extract_features, wav2vec_model_path=args.wav2vec_model_path, data_class='dev')
        features_label_test = run_extraction(extraction_func=extract_features, wav2vec_model_path=args.wav2vec_model_path, data_class='test')
        
        np.save(os.path.join(args.output_save, 'train_features.npy'), features_label_train)
        np.save(os.path.join(args.output_save, 'val_features.npy'), features_label_val)
        np.save(os.path.join(args.output_save, 'test_features.npy'), features_label_test)
    
    elif args.features_eval:
        features_label_eval = run_extraction(extraction_func=extract_features, wav2vec_model_path=args.wav2vec_model_path, data_class='eval')

        np.save(os.path.join(args.output_save, 'eval_features.npy'), features_label_eval)

    else:
        print('Command was not given, possible commands: --features_train, --features_eval')

if __name__ == '__main__':
    main()