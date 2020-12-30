from utils.config import Config

import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F
import utils.features_extraction as features_extraction
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from numpy import argmax

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def swish(x):
    return x * torch.sigmoid(x)

class Network(nn.Module):
    def __init__(self, input_size):
        super(Network, self).__init__()
        self.tc1 = nn.Linear(input_size, 4096)
        self.tc2 = nn.Linear(4096, 128)
        self.tc3 = nn.Linear(128, 4096)
        self.tc4 = nn.Linear(4096, 128)
        self.tc5 = nn.Linear(128, 128)
        self.tc6 = nn.Linear(128, 2)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.5)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.float()
        x = self.dropout1(F.relu(self.tc1(x)))
        x = self.dropout2(swish(self.tc2(x)))
        x = self.dropout3(swish(self.tc3(x)))
        x = self.dropout4(swish(self.tc4(x)))
        x = self.dropout5(swish(self.tc5(x)))
        x = self.tc6(x)

        return x

def model_accuracy(output, label):
    """Calculate the model accuracy"""
    
    pb = F.softmax(output, dim=1)
                
    _, top_class = pb.topk(1, dim=1)

    equals = label == top_class.view(-1)

    return torch.mean(equals.type(torch.FloatTensor))

def test_model(checkpoint_path, feature_path):
    """Test Model"""

    features_label_test = np.load(os.path.join(feature_path, 'test_features.npy'), allow_pickle=True)

    test_dataloader, input_size = construct_dataloader(checkpoint_path, 2, features_label_test)

    model = Network(input_size)
    model.to(device)

    # Load checkpointed model
    model.load_state_dict(torch.load(checkpoint_path))

    with torch.no_grad():
        model.eval()
        test_accuracy = 0

        for test_audio, test_label in test_dataloader:
            test_audio, test_label = test_audio.to(device), test_label.to(device)

            output = model(test_audio)

            test_accuracy += model_accuracy(output, test_label)

        else:
            print('Test Accuracy: {:.3f}'.format(test_accuracy/len(test_dataloader)))

def eval_model(checkpoint_path, feature_path):
    """Evaluate Model"""

    features_label_eval = np.load(os.path.join(feature_path, 'eval_features.npy'), allow_pickle=True)

    eval_dataloader, input_size = construct_dataloader(features_label_eval)

    model = Network(input_size)
    model.to(device)

    # Load checkpointed model
    model.load_state_dict(torch.load(checkpoint_path))

    with torch.no_grad():
        model.eval()
        results = []

        for eval_audio, eval_label in eval_dataloader:
            eval_audio, eval_label = eval_audio.to(device), eval_label.to(device)

            output = model(eval_audio)

            results.append(output)

        print(results)
    

def construct_dataloader(feature_path, num_workers, features_label):
    """Get data loader from data"""

    X, y = features_extraction.process_features_labels(features_label)

    y = argmax(y, axis=1) 

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=X.shape[0], 
                                num_workers=num_workers, pin_memory=True)
    
    return dataloader, X.shape[1]

def train_model(feature_path, checkpoint_output_path, num_workers, patience=25):
    """Train model given features"""

    features_label_train = np.load(os.path.join(feature_path, 'train_features.npy'), allow_pickle=True)
    features_label_val = np.load(os.path.join(feature_path, 'val_features.npy'), allow_pickle=True)
    features_label_test = np.load(os.path.join(feature_path, 'test_features.npy'), allow_pickle=True)

    train_dataloader, input_size = construct_dataloader(feature_path, num_workers, features_label_train)
    val_dataloader, _ = construct_dataloader(feature_path, num_workers, features_label_val)
    test_dataloader, _ = construct_dataloader(feature_path, num_workers, features_label_test)

    model = Network(input_size)
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

        for train_audio, train_label in train_dataloader:
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

                for val_audio, val_label in val_dataloader:
                    val_audio, val_label = val_audio.to(device), val_label.to(device)

                    out = model(val_audio)

                    loss = criterion(out, val_label)

                    val_accuracy += model_accuracy(out, val_label)

                    val_loss += loss.item()
            

            print('Epoch: {}/{} | '.format(e+1, epochs), 
            'Train Accuracy: {:.3f} | '.format((train_accuracy/len(train_dataloader))*100),
            'Val Accuracy: {:.3f} | '.format((val_accuracy/len(val_dataloader))*100),
            'Train Loss: {:.6f} | '.format(running_loss/len(train_dataloader)),
            'Val loss: {:.6f}'.format(val_loss/len(val_dataloader)))

            if loss < loss_min:
                print("\nValidation Loss Decreasead ({:.6f} --> {:.6f}), saving model...\n".format(loss_min, loss))
                loss_min = loss
                torch.save(model.state_dict(), checkpoint_output_path)
            
            else:
                p += 1

                if p == patience:
                    print("Early Stopping... ")
                    break
            
            model.train()
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=Config.base_dir) 
    parser.add_argument('--checkpoint_output', default=Config.checkpoint_output, help='Name of checkpoint file')   
    parser.add_argument('-i','--feature_dir', type=str, default=Config.feature_dir, help='Path to the features')   
    parser.add_argument('--checkpoint_path', default= Config.checkpoint_path_FNN, help="Checkpoint path to evaluate model")
    parser.add_argument("--num_workers", default=Config.num_workers, type=int, help="Number of workers")
    parser.add_argument("--patience", default=Config.patience, type=int, help="Pacience for early stopping")
    parser.add_argument("--train", help="Train Model", action="store_true")
    parser.add_argument("--test", help="Test Model", action="store_true")
    args = parser.parse_args()

    feature_path = os.path.join(args.base_dir, args.feature_dir)

    if args.train:
        checkpoint_path_out = os.path.join(args.base_dir, args.checkpoint_output)
        train_model(feature_path, checkpoint_path_out, args.num_workers, args.patience)
    
    elif args.test:
        checkpoint_path = os.path.join(args.base_dir, args.checkpoint_path)
        test_model(checkpoint_path, feature_path)

    else:
        print('Command was not given, possible commands: --train, --test')

if __name__ == '__main__':
    main()