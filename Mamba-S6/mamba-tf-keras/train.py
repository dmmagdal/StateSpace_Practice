


import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, Model

from dataclasses import dataclass
from einops import rearrange, repeat
from typing import Union

from transformers import AutoTokenizer

import datasets
import math
import numpy as np

from datasets import load_dataset
from tqdm import tqdm

from model import ModelArgs, init_model


# Initialize tokenizer (load from pretrained).
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size


# Initialize the model.
args = ModelArgs(
    model_input_dims=128,
    model_states=32,
    num_layers=12,
    dropout_rate=0.2,
    vocab_size=vocab_size,
    num_classes=1,
    loss='binary_crossentropy',
)
model = init_model(args)
model.summary()

# Load the dataset.
dataset = load_dataset("ajaykarthick/imdb-movie-reviews")

# Preprocess dataset into numpy and then tf.data.Dataset.
train_labels, test_labels = [], []
train_ids = np.zeros((len(dataset['train']), args.seq_length))
test_ids = np.zeros((len(dataset['test']), args.seq_length))

for i, item in enumerate(tqdm(dataset['train'])):
    text = item['review']
    train_ids[i, :] = tokenizer.encode_plus(
            text, 
            max_length=args.seq_length, 
            padding='max_length', 
            return_tensors='np')['input_ids'][0][:args.seq_length]

    train_labels.append(item['label'])

for i, item in enumerate(tqdm(dataset['test'])):
    text = item['review']
    test_ids[i, :] = tokenizer.encode_plus(
            text, 
            max_length=args.seq_length, 
            padding='max_length', 
            return_tensors='np')['input_ids'][0][:args.seq_length]

    test_labels.append(item['label'])

del dataset # delete the original dataset to save some memory

BATCH_SIZE = 32
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_ids, train_labels)
).batch(BATCH_SIZE).shuffle(1000)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_ids, test_labels)
).batch(BATCH_SIZE).shuffle(1000)

# Training loop.
history = model.fit(
    train_dataset, validation_data=test_dataset, epochs=10
)


# Run inference.
def infer(text: str, model: Model, tokenizer):
    tokens = tokenizer.encode(
            "Hello what is up", 
            max_length=args.seq_length, 
            padding='max_length', return_tensors='np')
    output = model(tokens)[0, 0]
    return output