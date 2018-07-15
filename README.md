# Implementation of "Guided Attention for Large Scale Scene Text Verification"

This is my implementation of "Guided Attention for Large Scale Scene Text Verification". The work is done while I am in Internship at Huawei. The paper can be found [here](https://arxiv.org/abs/1804.08588).

The model takes an image and a text string as input, and then outputs the probability of the text string being present in the image.

I am new to deep learning. Actually I reused a lot of code from a similar paper called "Attention-based Extraction of Structured Information from Street View Imagery", whose authors released their project on [GitHub](https://github.com/tensorflow/models/tree/master/research/attention_ocr).

Since the SVBM dataset is not public, I used FSNS dataset to train the model. For the "positive label", I just selected the ground truth label. For the "negative label", I randomly selected other labels (ground truth labels which belong to other images) as candidates.

## Requirements

1. Install the tensorflow library and others:

```
pip install tensorflow-gpu
pip install Pillow
```

2. At least 158GB of free disk space to download the FSNS dataset:

```
cd datasets
aria2c -c -j 20 -i fsns_urls.txt
cd ..
```

## Training

To train the model, just run:

```
python train.py
```

## Test

To test the model, just run:

```
python test.py
```
