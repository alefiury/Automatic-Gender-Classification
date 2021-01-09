# Gender-Classification
---
### Introduction

This is an implementation in pytorch of deep learning models for gender classification. 

There are three different models: Fully Connected, 1D Convolution and 2D Convolutional.

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

Settings can be changed in the **config.py** file.

To train the Convolutional 2D model, first, is necessary to create mel spectogram images from the audios.
Just run the folling script:

```
python CNN_2D.py.py --construct_images
```

Then, to finally train the models, you can just run:

For the Fully Connected model:

```
python FNN.py -train
```

For the 1D Convolutional:

```
python CNN_1D.py -train
```

For the 2D Convolutional:

```
python CNN_2D.py -train
```

### Contact
e-mail: alef_iury_c.c@discente.ufg.br
