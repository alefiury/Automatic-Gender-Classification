from utils.features_extraction import run_extraction, concatenate_features, concatenate_label
from utils.config import Config

import tensorflow as tf

import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
from numpy import argmax
import librosa
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataGenerator(Dataset):
    def __init__(self, files, labels):
        'Initialization'
        self.labels = labels
        self.files_list = files
        self.sample_rate = 16000
        self.sample_duration = 3

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, index):

        X, y = self.data_generation(index)

        return torch.from_numpy(X).type(torch.FloatTensor), y.long()

    def data_generation(self, index):

        sample, _ = librosa.load(self.files_list[index], res_type='kaiser_fast')
      
        if librosa.get_duration(sample, self.sample_rate) < float(self.sample_duration):
            tmp = np.zeros(int(self.sample_duration)*self.sample_rate)
            tmp[:len(sample)] = sample
        
        else:
            tmp = sample[:int(self.sample_duration)*self.sample_rate]
        tmp = librosa.util.normalize(tmp)
        
        return tmp, self.labels[index]

class CNN_Network(nn.Module):
    def __init__(self):
        super(CNN_Network, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=5, stride=1)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=5, stride=1)

        self.maxpool = nn.MaxPool1d(2, 2)

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(512)
        self.batch_norm5 = nn.BatchNorm1d(1024)

    def forward(self, x):

        x = self.batch_norm1( self.maxpool( F.relu( self.conv1(x) ) ) )
        x = self.batch_norm2( self.maxpool( F.relu( self.conv2(x) ) ) )
        x = self.batch_norm3( self.maxpool( F.relu( self.conv3(x) ) ) )
        x = self.batch_norm4( self.maxpool( F.relu( self.conv4(x) ) ) )
        x = self.batch_norm5( self.maxpool( F.relu( self.conv5(x) ) ) )

        x = x.mean(2)

        x = self.dropout( F.relu( self.fc1(x) ) )
        x = self.dropout( F.relu( self.fc2(x) ) )
        x = self.fc3(x)

        return x

def model_accuracy(output, label, yn):
    
    pb = F.softmax(output, dim=1)
                
    _, top_class = pb.topk(1, dim=1)

    equals = label == top_class.view(-1)

    return torch.mean(equals.type(torch.FloatTensor))

def train_model(batch_size, num_workers, patience):

    train_dataloader = get_data_loader(batch_size, num_workers, 'train')
    val_dataloader = get_data_loader(batch_size, num_workers, 'dev')
    test_dataloader = get_data_loader(batch_size, num_workers, 'test')

    model = CNN_Network()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07)

    epochs = 120
    loss_min = np.Inf
    p = 0

    model.train()

    for e in range(epochs):

        running_loss = 0
        train_accuracy = 0

        for train_audio, train_label in tqdm(train_dataloader):
            train_audio = train_audio.view(train_audio.shape[0], 1, train_audio.shape[1])
            train_audio, train_label = train_audio.to(device), train_label.to(device)

            optimizer.zero_grad()
            
            out = model(train_audio)

            train_loss = criterion(out, train_label)

            train_loss.backward()

            optimizer.step()

            running_loss += train_loss

            train_accuracy += model_accuracy(out, train_label, False)

        else:

            val_loss = 0
            val_accuracy = 0

            with torch.no_grad():
                model.eval()

                for val_audio, val_label in tqdm(val_dataloader):
                    val_audio = val_audio.view(val_audio.shape[0], 1, val_audio.shape[1])
                    val_audio, val_label = val_audio.to(device), val_label.to(device)

                    out = model(val_audio)

                    loss = criterion(out, val_label)

                    val_accuracy += model_accuracy(out, val_label, True)

                    val_loss += loss
            

            print('Epoch: {}/{} | '.format(e+1, epochs), 
            'Train Accuracy: {:.3f} | '.format((train_accuracy/len(train_dataloader))*100),
            'Val Accuracy: {:.3f} | '.format((val_accuracy/len(val_dataloader))*100),
            'Train Loss: {:.6f} | '.format(running_loss/len(train_dataloader)),
            'Val loss: {:.6f}'.format(val_loss/len(val_dataloader)))

            if loss < loss_min:
                print("\nValidation Loss Decreasead ({:.6f} --> {:.6f}), saving model...\n".format(loss_min, loss))
                loss_min = loss
                torch.save(model.state_dict(), './outputs/1D_CNN_checkpoint.pth')
            
            else:
                p += 1

                if p == patience:
                    print("Early Stopping... ")
                    break
            
            model.train()

def test_model(batch_size, checkpoint_path, num_workers):
    test_dataloader = get_data_loader(batch_size, num_workers, 'test')

    model = CNN_Network()
    model.to(device)

    model.load_state_dict(torch.load(checkpoint_path))

    with torch.no_grad():
        model.eval()
        test_accuracy = 0

        for test_audio, test_label in tqdm(test_dataloader):
            test_audio = test_audio.view(test_audio.shape[0], 1, test_audio.shape[1])
            test_audio, test_label = test_audio.to(device), test_label.to(device)

            output = model(test_audio)

            test_accuracy += model_accuracy(output, test_label, False)

        else:
            print('Test Accuracy: {:.3f}'.format(test_accuracy/len(test_dataloader)))

def extract_files(line, model, data_class):   

    file = line['file']
    label = int(line['label'])
    filename = file.split('.')[0]

    folder = os.path.join(Config.base_dir, 'data', data_class, 'male') if label == 1 else os.path.join(Config.base_dir, 'data', data_class, 'female')

    filepath = os.path.join(folder, str(file))
    return filepath, label 

def prepare_X_y(data_class):
    features_label = run_extraction(extract_files, None, data_class)

    file_paths = concatenate_features(features_label)

    labels = concatenate_label(features_label)

    # Setting our X as a numpy array to feed into the neural network
    X = np.array(file_paths)

    # Setting our y
    y = np.array(labels)

    # Hot encoding y
    lb = LabelEncoder()
    y = tf.keras.utils.to_categorical(lb.fit_transform(y))

    return X, y

def get_data_loader(batch_size, num_workers, data_class):
    X, y = prepare_X_y(data_class)

    y = argmax(y, axis=1) 
    y = torch.from_numpy(y)


    dataset = DataGenerator(X, y) 
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, 
                                num_workers=num_workers, pin_memory=True)

    return dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=Config.batch_size_1D, help='Batch size')     
    parser.add_argument('-c','--checkpoint_path', default=Config.checkpoint_path_1D, help='Checkpoint path to evaluate model')
    parser.add_argument('-n', '--num_workers', default=Config.num_workers, type=int, help='Number of workers')
    parser.add_argument('--patience', default=Config.patience, type=int, help='Pacience for early stopping')
    parser.add_argument('--train', help='Train Model', action='store_true')
    parser.add_argument('--test', help='Evaluate Model', action='store_true')
    args = parser.parse_args()

    if args.train:
        train_model(args.batch_size, args.num_workers, args.patience)
    
    elif args.test:
        test_model(args.batch_size, args.checkpoint_path, args.num_workers)

    else:
        print('Command was not given, possible commands: --train, --eval')

if __name__ == '__main__':
    main()


