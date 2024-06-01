# This code runs experiments with the CLASSP optimizer for continual learning in Sentiment Analysis
# In case of using this software package or parts of it, cite:
# Oswaldo Ludwig, "CLASSP: a Biologically-Inspired Approach to Continual Learning through Adjustment Suppression and Sparsity Promotion", ArXiv, 2024.

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchtext import datasets
from torchtext.data import Field, LabelField
from torchtext.data import BucketIterator
from torch import nn, optim
from CLASSP import CLASSP_optimizer
from torchtext.datasets import YelpReviewFull
from torchtext.data import Field, TabularDataset
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchtext.data import Example, Dataset

import time
import spacy
import random
import numpy as np

batch_size = 750 # Changing this value impacts training performance, if you don't have enough VRAM, use gradient accumulation

nlp = spacy.load('en_core_web_sm')

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    OBS.: limiting sentences to 30 words to save VRAM since for the Transformer RAM consuption increases with the square of the input length
    """
    return [tok.text for tok in nlp.tokenizer(text)][0:30]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE = " + str(device), flush=True)

TEXT = Field(tokenize = tokenize_en, include_lengths = True)
LABEL = LabelField(dtype = torch.float)

from datasets import load_dataset

# Load the Financial Phrasebank dataset:
dataset = load_dataset("financial_phrasebank", "sentences_50agree")

# Access the train split
Train_data1 = dataset['train']

# Convert Hugging Face Dataset to torchtext Dataset
def hf_to_torchtext(hf_dataset, text_field, label_field):

    for n, hf_example in enumerate(hf_dataset):
        print("EXAMPLE " + str(n) + ": ")
        print(type(hf_example), hf_example)

    examples = []

    # Iterate over all examples in the Hugging Face Dataset
    for hf_example in hf_dataset:
        # Create a torchtext Example for this example
        example = Example.fromlist([hf_example['sentence'], hf_example['label']], fields=[('text', text_field), ('label', label_field)])
        # Add the torchtext Example to the list
        examples.append(example)

    # Create a torchtext Dataset from the examples and return it
    return Dataset(examples, fields=[('text', text_field), ('label', label_field)])

# Convert the Hugging Face Datasets to torchtext Datasets
Train_data1 = hf_to_torchtext(Train_data1, TEXT, LABEL)
print("Financial Phrasebank dataset loaded...", flush=True)

# Loading the IMDB dataset
print("Loading the IMDB dataset...", flush=True)
Train_data2, Test_data2 = datasets.IMDB.splits(TEXT, LABEL, root='/mnt/users/oswaldo_ludwig/catastrophic_forgetting')
print("IMDB loaded...", flush=True)

label_to_int = {'neg': 0, 'pos': 1, 0: 0, 1: 1, 2: 2}

def tokenize(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return text.split()

def build_vocab(tokenized_texts):
    """
    Builds a vocabulary from a list of tokenized texts
    """
    vocab = {'<PAD>': 0, '<UNK>': 1}  # PAD and UNK tokens
    for tokens in tokenized_texts:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def convert_to_indexes(tokens, vocab):
    """
    Converts a list of tokens to a list of indexes based on the vocabulary
    """
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

texts = [" ".join(example.text) for example in Train_data1]
texts += [" ".join(example.text) for example in Train_data2]

tokenized_texts = [tokenize(text) for text in texts]
vocabulary = build_vocab(tokenized_texts)
print("Vocabulary: ", flush=True)
print(vocabulary, flush=True)
num_examples_dataset1 = len(Train_data1.examples)

# Defining the model:
class MyModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_head, output_dim, n_layers,
                 dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head)
        self.self_attn = TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        embedded = self.layer_norm(embedded)
        attention_output = self.self_attn(embedded)
        attention_output = self.layer_norm(attention_output)
        hidden = attention_output[-1]

        return self.fc(hidden.squeeze(0))

# Defining a function to calculate accuracy:
def calculate_accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            texts, labels = data
            Inputs = [torch.tensor(convert_to_indexes(seq, vocabulary)) for seq in texts]
            Inputs = nn.utils.rnn.pad_sequence(Inputs, padding_value=pad_idx)
            text_lengths = Inputs.ne(pad_idx).sum(dim=0)
            # Sort lengths in decreasing order
            sorted_lengths, sorted_idx = text_lengths.sort(descending=True)
            # Sort Inputs according to the sorted indices
            sorted_inputs = Inputs[:, sorted_idx]
            # Convert labels to a Tensor and sort them according to the sorted indices
            labels = torch.tensor([label_to_int[label] for label in labels])
            sorted_labels = labels[sorted_idx]
            outputs = model(sorted_inputs.to(device), sorted_lengths)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == sorted_labels.to(device)).sum().item()
    return correct / total

train_data1 = list(Train_data1)
train_data2 = list(Train_data2)

def collate_fn(batch):
    # Check the type of the first item in the batch
    if isinstance(batch[0], tuple):
        # If the items are tuples, assume they are (label, text)
        inputs = [item[1] for item in batch]
        labels = [item[0] for item in batch]
    else:
        # If the items are dicts, assume they have 'text' and 'label' keys
        inputs = [item.text for item in batch]
        labels = [item.label for item in batch]

    return inputs, labels

def one_hot_encode_list(lst):
    # Create a tensor of zeros with size (len(lst), )
    one_hot = torch.zeros(len(lst), output_dim)
    # Set the nth element of each row to 1
    for i, n in enumerate(lst):
        one_hot[i][n] = 1

    return one_hot

# Training function
def train(customOptimizer, Shuffle, LR=0.001, Threshold=0.2, Epsilon=1e-4, Power=1):
  print("Creating a dataloader...", flush=True)
  if Shuffle==False:
     text = "without shuffling and "
     dataloader1 = DataLoader(train_data1, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
     dataloader2 = DataLoader(train_data2, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
  else:
     text = "shuffling and "
     dataloader1 = DataLoader(train_data1, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
     dataloader2 = DataLoader(train_data2, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
  if customOptimizer==False:
     optimizer = optim.SGD(model.parameters(), lr=LR)
     #optimizer = torch.optim.Adam(model.parameters(), lr=LR)

     text += "standard optimizer"
  else:
     optimizer = CLASSP_optimizer(model.parameters(), lr=LR, threshold=Threshold, epsilon=Epsilon, power=Power)  #  sigma controls the decay rate of the weight update
     text += "custom optimizer"

  print("Training on first dataset...", flush=True)
  time_sec = time.time()
  counter = 0
  loss = 1e10
  while loss > 0.005:
   for inputs, labels in dataloader1:
    counter += 1
    print("counter = " + str(counter), flush=True)
    Inputs = [torch.tensor(convert_to_indexes(seq, vocabulary)) for seq in inputs]
    Inputs = nn.utils.rnn.pad_sequence(Inputs, padding_value=pad_idx)
    text_lengths = Inputs.ne(pad_idx).sum(dim=0)
    # Sort lengths in decreasing order
    sorted_lengths, sorted_idx = text_lengths.sort(descending=True)
    # Sort Inputs according to the sorted indices
    sorted_inputs = Inputs[:, sorted_idx]
    # Convert labels to a Tensor and sort them according to the sorted indices
    labels = torch.tensor([label_to_int[label] for label in labels])
    sorted_labels = labels[sorted_idx]
    Labels = one_hot_encode_list(sorted_labels)
    Labels = torch.tensor(Labels)
    optimizer.zero_grad()
    outputs = model(sorted_inputs.to(device), sorted_lengths)  # Pass the sorted inputs and lengths to the model
    loss = criterion(outputs, Labels.to(device))
    loss.backward()
    print("loss = " + str(loss), flush=True)

    if customOptimizer==True:
        average_grad_sum = optimizer.average_grad_sum()
        print("average_grad_sum = " + str(average_grad_sum), flush=True)
        optimizer.step(lr=LR, threshold=Threshold, epsilon=Epsilon, power=Power, apply_scaling=True)

    else:
        optimizer.step()

 # Calculating and printing the accuracy on both datasets
  test_dataloader1 = DataLoader(train_data1, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
  accuracy01 = calculate_accuracy(model, test_dataloader1)
  print('Accuracy on dataset#1 ' + text + ': ' + str(accuracy01*100) + '%')
  print("loss first dataset = " + str(loss), flush=True)
  print("Elapsed time: " + str(time.time() - time_sec) + " seconds", flush=True)
  print("Training on second dataset...", flush=True)
  loss = 1e10

  # Change this threshold in accordancy with the optimizer to keep acc on IMDB in about 85%
  # and ensure a fair comparison, as overfitting IMDB means forgetting the first domain (around 0.28 for CLASSP, 0.48 for Adam and 0.32 for SGD) 
  while loss > 0.28:
   for inputs, labels in dataloader2:
    counter += 1
    Inputs = [torch.tensor(convert_to_indexes(seq, vocabulary)) for seq in inputs]
    Inputs = nn.utils.rnn.pad_sequence(Inputs, padding_value=pad_idx)
    text_lengths = Inputs.ne(pad_idx).sum(dim=0)
    # Sort lengths in decreasing order
    sorted_lengths, sorted_idx = text_lengths.sort(descending=True)
    # Sort Inputs according to the sorted indices
    sorted_inputs = Inputs[:, sorted_idx]
    # Convert labels to a Tensor and sort them according to the sorted indices
    labels = torch.tensor([label_to_int[label] for label in labels])
    sorted_labels = labels[sorted_idx]
    Labels = one_hot_encode_list(sorted_labels)
    Labels = torch.tensor(Labels)
    optimizer.zero_grad()
    outputs = model(sorted_inputs.to(device), sorted_lengths)  # Pass the sorted inputs and lengths to the model
    loss = criterion(outputs, Labels.to(device))
    loss.backward()
    print("loss = " + str(loss), flush=True)

    if customOptimizer==True:
        average_grad_sum = optimizer.average_grad_sum()
        print("average_grad_sum = " + str(average_grad_sum), flush=True)
        optimizer.step(lr=LR, threshold=Threshold, epsilon=Epsilon, power=Power, apply_scaling=True)
    else:
        optimizer.step()

  print("loss second dataset = " + str(loss), flush=True)
  print("Elapsed time: " + str(time.time() - time_sec) + " seconds", flush=True)

  # Calculating and printing the accuracy on both datasets
  test_dataloader1 = DataLoader(train_data1, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
  test_dataloader2 = DataLoader(train_data2, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
  accuracy1 = calculate_accuracy(model, test_dataloader1)
  accuracy2 = calculate_accuracy(model, test_dataloader2)
  return(accuracy1*100, accuracy2*100, accuracy01*100)

# Model parameters
vocab_size = len(vocabulary)  # According to TEXT field preprocessing
embedding_dim = 600  # ATTENTION: it must be a multiple of n_head
output_dim = 3  # For including Financial Phasebank classification, which has 3 classes (positive, neutral and negative)
n_layers = 1  # The complexity of this task requires only one layer
n_head = 30  # Check the compliance with embedding_dim
dropout = 0.1 # Small value because we are not investigating generalization capacity here
pad_idx = 0  # According to TEXT field preprocessing


print("Training...", flush=True)
criterion = nn.CrossEntropyLoss()

# Initialize the model
model = MyModel(vocab_size, embedding_dim, n_head, output_dim, n_layers, dropout, pad_idx)
print("Model object instantiated...")
model.to(device)
print("Model to device...")
acc1, acc2, acc01 = train(True, True, LR=0.025, Threshold=0.99, Epsilon=1e-4, Power=1) # optimal CLASSP
#acc1, acc2, acc01 = train(True, True, LR=0.025, Threshold=0, Epsilon=1e-4, Power=1) # CLASSP ablation study: no threshold
#acc1, acc2, acc01 = train(True, True, LR=0.025, Threshold=0, Epsilon=1e-4, Power=2) # CLASSP ablation study: no threshold + p=2 (Adagrad instance)
#acc1, acc2, acc01 = train(False, True, LR=0.0003) # optimal Adam optimizer
#acc1, acc2, acc01 = train(False, True, LR=0.025) # optimal SGD optimizer

print('Accuracy on dataset#1 after fine-tuning on dataset#2: ' + str(acc1) + '%')
print('Accuracy on dataset#2 (controlled): ' + str(acc2) + '%')
print('Accuracy on dataset#1 before fine-tuning on dataset#2: ' + str(acc01) + '%')
