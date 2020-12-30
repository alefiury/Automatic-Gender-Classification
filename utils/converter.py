import argparse
import os 
import shutil
import glob
import tqdm
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import math
from pathlib import Path

from config import Config

from os.path import join, exists, isdir, getsize, dirname

def analyse_data(files_path, dataset_type, folder_name):   
    metadata_path = glob.glob(files_path + '/**/*' + 'SPEAKERS.TXT', recursive=True)[0]
    with open(metadata_path, 'r') as read_obj:
        for line in tqdm.tqdm(read_obj):
            if not line.startswith(';'):
                id, sex, subset, minutes, name = map(str.strip, line.split(' | '))
                if subset == dataset_type:
                    move_data(files_path, sex, id, dataset_type, folder_name)

def move_data(files_path, sex, id, dataset_type, folder_name):
    if sex == 'M':
        gender_dir = 'male'
    elif sex == 'F':
        gender_dir = 'female'

    output_path = os.path.join(Config.base_dir, folder_name, gender_dir)

    os.makedirs(output_path, exist_ok=True)

    for file_path in glob.glob(os.path.join(files_path, '**', id, '**', '*.wav'), recursive=True):
        shutil.move(file_path, os.path.join(output_path, file_path.split('/')[-1]))

def convert2wav(files_path, in_extension):
    """
    Convert audio to wav 16khz, 16 bits signed-integer
    """

    for file_path in tqdm.tqdm(glob.glob(files_path + '/**/*' + in_extension, recursive=True)):

        # Directory of output
        output_path = dirname(file_path.replace(file_path.split('/')[-1], ''))

        # New file name with .wav as new extension
        new_filename = file_path.split('/')[-1].replace(in_extension, '') + '.wav'

        # Load audio in 16khz, sample rate conversion
        song, sr = librosa.load(file_path, sr=16000)
        # Write audio in 16khz, 16 bits and signed-integer
        sf.write(os.path.join(output_path, new_filename), song, sr, subtype='PCM_16')

        # Remove old file from directory
        os.remove(file_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files_path', 
                        help='File path to audios',
                        type=str) 

    parser.add_argument('--in_extension', 
                        help='Extension of the original audios to be converted',
                        type=str) 

    parser.add_argument('--dataset_type', 
                        help='Dataset type - train-clean-100, dev-clean, test-clean, eval',
                        type=str)

    parser.add_argument('--folder_name', 
                        help='Dataset type - train, dev, test, eval',
                        type=str)

    parser.add_argument('-c', 
                        '--convert2wav', 
                        help='Command to convert audio files', 
                        action='store_true')
    
    parser.add_argument('-m', 
                        '--move_files', 
                        help='Command to move audio files', 
                        action='store_true')

                        
    args = parser.parse_args()

    if args.convert2wav:
        convert2wav(args.files_path, args.in_extension)

    elif args.move_files:
        analyse_data(args.files_path, args.dataset_type, args.folder_name)


    else:
        print('Command was not given, possible commands: --convert2wav, --verify_sample_rate_waves')


if __name__ == '__main__':
    main()
