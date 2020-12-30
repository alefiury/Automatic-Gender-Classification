import utils.process_data as process_data
from utils.config import Config

import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import librosa
import librosa.display

import os
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def images(files, data_type):
    """adapted from jurgenarias's function: https://github.com/jurgenarias/Portfolio/blob/master/Voice%20Classification/Code/Gender_Classifier/Gender_Classifier_CNN.ipynb"""

    sample_duration = 1

    file = files.file
    label = files.label
    gender = 'male' if label == '1' else 'female'

    folder = os.path.join('./data', data_type, gender)
    
    filepath = os.path.join(folder, str(file.split('/')[1]))
    
    # Loading the image with no sample rate to use the original sample rate and
    # kaiser_fast to make the speed faster according to a blog post about it (on references)
    X, sample_rate = librosa.load(filepath, res_type='kaiser_fast')

    sample = X

    if librosa.get_duration(sample, sample_rate) < sample_duration:
      sample = np.pad(sample, sample_duration - len(sample))
    else:
      sample = sample[:sample_duration*sample_rate]
   
    # Setting the size of the image
    fig = plt.figure(figsize=[2,4])
    
    # This is to get rid of the axes and only get the picture 
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    
    # This is the melspectrogram from the decibels with a linear relationship
    # Setting min and max frequency to account for human voice frequency
    S = librosa.feature.melspectrogram(y=sample, sr=sample_rate, n_fft=512)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', fmin=50, fmax=280)
    
    # Here we choose the path and the name to save the file, we will change the path when
    # using the function for train, val and test to make the function easy to use and output
    # the images in different folders to use later with a generator
    name = files.file
    output_path = os.path.join('./', 'data', data_type, 'images', str(name).split('/')[0], gender)
    os.makedirs(output_path, exist_ok=True)
    file  =  os.path.join(output_path, str(name).split('/')[1] + '.jpg')
    # print(file)
    
    # Here we finally save the image file choosing the resolution 
    plt.savefig(file, dpi=500, bbox_inches='tight',pad_inches=0)
    
    # Here we close the image because otherwise we get a warning saying that the image stays
    # open and consumes memory
    plt.close()

def make_jpg_train(files):
    return 'train/' + str(files)

def make_jpg_test(files):
    return 'test/' + str(files)

def make_jpg_val(files):
    return 'val/' + str(files)

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
    plt.show()

class CNN_Network(nn.Module):
    def __init__(self):
        super(CNN_Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1)

        self.maxpool = nn.MaxPool2d(2, 2)


        self.fc1 = nn.Linear(256 * 6 * 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

        self.dropout = nn.Dropout(0.5)

        # self.adpt = nn.AdaptiveAvgPool2d((2, 1))

        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        # self.batch_norm5 = nn.BatchNorm2d(512)
    
    def forward(self, x):
        # print(x.shape)

        x = self.batch_norm1( self.maxpool( F.relu( self.conv1(x) ) ) )
        # print(x.shape)
        
        x = self.batch_norm2( self.maxpool( F.relu( self.conv2(x) ) ) )
        # print(x.shape)
        
        x = self.batch_norm3( self.maxpool( F.relu( self.conv3(x) ) ) )
        # print(x.shape)
        
        x = self.batch_norm4( self.maxpool( F.relu( self.conv4(x) ) ) )

        x = x.view(-1, 256 * 6 * 2)
        x = self.dropout(x)
        # print(x.shape)

        x = self.dropout( F.relu( self.fc1(x) ) )
        
        x = self.dropout( F.relu( self.fc2(x) ) )
        
        x = self.fc3(x)
        
        return x

def test_model(batch_size, checkpoint_path, num_workers):
    _, _, test_dataloader = get_data_loader(batch_size, num_workers)

    model = CNN_Network()
    model.to(device)

    model.load_state_dict(torch.load(checkpoint_path))

    with torch.no_grad():
        model.eval()
        test_accuracy = 0

        for test_audio, test_label in tqdm.tqdm(test_dataloader):
            test_audio, test_label = test_audio.to(device), test_label.to(device)

            output = model(test_audio)

            test_accuracy += model_accuracy(output, test_label)

        else:
            print('Test Accuracy: {:.3f}'.format(test_accuracy/len(test_dataloader)))

def construct_images():

    tqdm.tqdm.pandas()

    train = process_data.concatenate_dfs('train')
    val = process_data.concatenate_dfs('dev')
    test = process_data.concatenate_dfs('test')

    train['file'] = train['file'].progress_apply(make_jpg_train)
    val['file'] = val['file'].progress_apply(make_jpg_test)
    test['file'] = test['file'].progress_apply(make_jpg_val)

    train.progress_apply(images, args=('train',), axis=1)
    test.progress_apply(images, args=('dev',), axis=1)
    val.progress_apply(images, args=('test',), axis=1)

def get_data_loader(batch_size, num_workers):
    
    tqdm.tqdm.pandas()

    train = process_data.concatenate_dfs('train')
    val = process_data.concatenate_dfs('dev')
    test = process_data.concatenate_dfs('test')

    train['file'] = train['file'].progress_apply(make_jpg_train)
    val['file'] = val['file'].progress_apply(make_jpg_test)
    test['file'] = test['file'].progress_apply(make_jpg_val)

    train_transforms = transforms.Compose([ transforms.Resize(64),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                            ])

    test_transforms = transforms.Compose([  transforms.Resize(64),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

    val_transforms = transforms.Compose([   transforms.Resize(64),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

    output_path = os.path.join('./', 'data')

    train_data = datasets.ImageFolder(os.path.join(output_path, 'train', 'images'), transform=train_transforms)
    test_data = datasets.ImageFolder(os.path.join(output_path, 'test', 'images'), transform=test_transforms)
    val_data = datasets.ImageFolder(os.path.join(output_path, 'dev', 'images'), transform=val_transforms)

    train_dataloader = torch.utils.data.DataLoader(train_data, 
                                                    batch_size=batch_size, 
                                                    num_workers=num_workers,
                                                    shuffle = True)

    test_dataloader = torch.utils.data.DataLoader(test_data, 
                                                batch_size=batch_size,
                                                num_workers=num_workers)

    val_dataloader = torch.utils.data.DataLoader(val_data, 
                                                batch_size=batch_size,
                                                num_workers=num_workers)

    return train_dataloader, test_dataloader, val_dataloader

def model_accuracy(output, label):
    
    pb = F.softmax(output, dim=1)
                
    _, top_class = pb.topk(1, dim=1)

    equals = label == top_class.view(-1)

    return torch.mean(equals.type(torch.FloatTensor))

def train_model(batch_size, num_workers, patience):

    train_dataloader, val_dataloader, _ = get_data_loader(batch_size, num_workers)

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

        for train_audio, train_label in tqdm.tqdm(train_dataloader):
            train_audio, train_label = train_audio.to(device), train_label.to(device)

            optimizer.zero_grad()
            
            out = model(train_audio)

            train_loss = criterion(out, train_label)

            train_loss.backward()

            optimizer.step()

            running_loss += train_loss

            train_accuracy += model_accuracy(out, train_label)

        else:

            val_loss = 0
            val_accuracy = 0

            with torch.no_grad():
                model.eval()

                for val_audio, val_label in tqdm.tqdm(val_dataloader):
                    val_audio, val_label = val_audio.to(device), val_label.to(device)

                    out = model(val_audio)

                    loss = criterion(out, val_label)

                    val_accuracy += model_accuracy(out, val_label)

                    val_loss += loss
            

            print('Epoch: {}/{} | '.format(e+1, epochs), 
            'Train Accuracy: {:.3f} | '.format((train_accuracy/len(train_dataloader))*100),
            'Val Accuracy: {:.3f} | '.format((val_accuracy/len(val_dataloader))*100),
            'Train Loss: {:.6f} | '.format(running_loss/len(train_dataloader)),
            'Val loss: {:.6f}'.format(val_loss/len(val_dataloader)))

            if loss < loss_min:
                print("\nValidation Loss Decreasead ({:.6f} --> {:.6f}), saving model...\n".format(loss_min, loss))
                loss_min = loss
                torch.save(model.state_dict(), './outputs/2D_CNN_checkpoint.pth')
            
            else:
                p += 1

                if p == patience:
                    print('Early Stopping ...')
                    break
            
            model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=Config.base_dir) 
    parser.add_argument('-b', '--batch_size', type=int, default=Config.batch_size_2D, help='Batch size')     
    parser.add_argument('-c','--checkpoint_path', default=Config.checkpoint_path_2D, help='Checkpoint path to evaluate model')
    parser.add_argument('-n', '--num_workers', default=Config.num_workers, type=int, help='Number of workers')
    parser.add_argument('--patience', default=Config.patience, type=int, help='Pacience for early stopping')
    parser.add_argument('--construct_images', help='Construct melspectrogram images', action='store_true')
    parser.add_argument('--train', help='Train Model', action='store_true')
    parser.add_argument('--eval', help='Evaluate Model', action='store_true')
    args = parser.parse_args()

    if args.train:
        train_model(args.batch_size, args.num_workers, args.patience)
    
    elif args.eval:
        test_model(args.batch_size, args.checkpoint_path, args.num_workers)

    elif args.construct_images:
        construct_images()


    else:
        print('Command was not given, possible commands: --train, --eval')

if __name__ == '__main__':
    main()

