# Gender-Classification

### Introduction

This is an implementation in pytorch of deep learning models for gender classification. 

There are three different models: Fully Connected, Convolution 1D and Convolutional 2D.

### Data
The Librispeech corpus was used.
**The
following script downloads and prepares the data**:

```
./download_clean_datasets.sh
```


Or alternatively download the dataset, convert the wav files to 16kHz PCM 16 bits
and set the 'train', 'clean' and 'dev' directories.

### Training
To train the Fully Connected model, first, is necessary to create the features. Run the fullowing script:

```
python FNN_features_extraction.py -t
```

After the feature creation you can simply run the training by:

```
python FNN.py -train
```

Settings can be changed in the **config.py** file.

To train the other two models, you can just follow the FC example:

```
python CNN_1D.py -train
```

and

```
python CNN_2D.py -train
```
