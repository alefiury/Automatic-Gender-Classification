# Automatic Gender Classification

- [Introduction](#Introduction)
- [Data](#Data)
- [Training](#Training)
- [Citation](#Citation)
- [Contact](#Contact)

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
@INPROCEEDINGS{219528,
    AUTHOR="Alef Iury Siqueira Ferreira and Frederico de Oliveira and Nadia Silva and Anderson Soares",
    TITLE="A Comparison of Deep Learning Architectures for Automatic Gender Recognition from Audio Signals",
    BOOKTITLE="ENIAC 2021 () ",
    ADDRESS="",
    DAYS="29-3",
    MONTH="nov",
    YEAR="2021",
    ABSTRACT="Automatic gender recognition from speech is a problem related to the area of speech analysis and has a variety of applications thats extends from the personalisation of product recommendation to forensics. Identifying the efficiency and costs of different approaches that deal with this problem is imperative. This work aims to investigate and compare the efficiency and costs of different deep learning architectures in the task of gender recognition from speech. The results show that the one-dimensional convolutional model achieves the best results. However, experiments conducted demonstrate that the fully connected model has similar results, using less memory and trained in much less time compared to the one-dimensional convolutional model.",
    KEYWORDS="Applications of Artificial Intelligence; Artificial Neural Networks; Deep Learning; Machine Learning",
    URL="http://XXXXX/219528.pdf"
}
```

# Contact
e-mail: alef_iury_c.c@discente.ufg.br
