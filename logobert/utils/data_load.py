from sentence_transformers import InputExample
import torch
import csv

def load_train_objs(train_filepath):
    train_samples = []
    with open(train_filepath, 'r', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
            train_samples.append(InputExample(texts=[row['query'], row['text']], label=float(row['label'])))
            train_samples.append(InputExample(texts=[row['text'], row['query']], label=float(row['label'])))
    return train_samples


def load_val_objs(dev_filepath):
            dev_samples = []
            with open(dev_filepath, 'r', encoding='utf8') as fIn:
                reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_NONE) #, 기준으로 나눔. csv --> dict
                for row in reader:
                    dev_samples.append(InputExample(texts=[row['query'], row['text']], label=float(row['label'])))
            return dev_samples

