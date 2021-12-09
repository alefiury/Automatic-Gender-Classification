# Automatic Gender Classification

- [Introduction](#Introduction)
- [Data](#Data)
- [Training](#Training)
- [Citation](#Citation)
- [Contact](#Contact)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

# Introduction

This is an implementation in Pytorch of deep learning models for automatic gender classification for the paper:
A Comparison of Deep Learning Architectures for AutomaticGender Recognition from Audio Signals.

There are three different models: Fully Connected, Unidimensional Convolutional Neural Network (1D CNN) and Bidimensional Convolutional Neural Network (2D CNN).

# Data

The [Librispeech](https://www.openslr.org/12) corpus is used.

**Run the following bash command to download and prepare the data**:

```
./download_clean_datasets.sh
```

Or alternatively download the dataset that you prefer, convert the wav files to 16kHz PCM 16 bits
and set the 'train', 'clean' and 'dev' directories.


# Training

To train the Fully Connected model, first, is necessary to extract the features. 

To extract and save the audio features, run the following script:

```
python FNN_features_extraction.py -t
```

To train the Convolutional 2D model is necessary to create mel spectogram images.
Just run the following script:

```
python CNN_2D.py.py --construct_images
```

Then, to train the models, you can just run:

- For the Fully Connected model:

```
python FNN.py -train
```

- For the 1D Convolutional:

```
python CNN_1D.py -train
```

- For the 2D Convolutional:

```
python CNN_2D.py -train
```

# Configuration

The relevant information related with the training configuration can be found and changed in the `config.py` file, inside the `utils` folder.

# Citation

If you use this code for your research, please consider citing:

```
@inproceedings{eniac,
 author = {Alef Ferreira and Frederico Oliveira and Nádia Silva and Anderson Soares},
 title = {A Comparison of Deep Learning Architectures for Automatic Gender Recognition from Audio Signals},
 booktitle = {Anais do XVIII Encontro Nacional de Inteligência Artificial e Computacional},
 location = {Evento Online},
 year = {2021},
 keywords = {},
 issn = {0000-0000},
 pages = {715--726},
 publisher = {SBC},
 address = {Porto Alegre, RS, Brasil},
 url = {https://sol.sbc.org.br/index.php/eniac/article/view/18297}
}
```

# Contact
e-mail: alef_iury_c.c@discente.ufg.br
