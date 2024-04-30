# This code runs experiments with the CLASSP optimizer for continual learning
# In case of using this software package or parts of it, cite:
# Oswaldo Ludwig, "CLASSP: a Biologically-Inspired Approach to Continual Learning through Adjustment Suppression and Sparsity Promotion", ArXiv, 2024.

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn, optim
from CLASSP import CLASSP_optimizer
import time

n_experiments = 1

# Defining model:
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

# Defining a function to calculate accuracy:
def calculate_accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


print("Loading the datasets...", flush=True)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)

num_examples_dataset1 = len(dataset1)

print("MNIST loaded...", flush=True)
dataset2 = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
print("FashionMNIST loaded...", flush=True)

print("Combining the datasets without shuffling...", flush=True)
combined_dataset = ConcatDataset([dataset1, dataset1, dataset1, dataset1, dataset2]) # For 4 training epochs in the first dataset and then fine tuning one epoch with the second datset

def train(customOptimizer, Shuffle, LR=0.2, Threshold=0.5, Epsilon=1e-5, Power=1):
  print("Creating a dataloader...", flush=True)
  if Shuffle==False:
     text = "without shuffling (i.e. subject to catastrophic forgetting) and "
     dataloader = DataLoader(combined_dataset, batch_size=32, shuffle=False)
  else:
     text = "shuffling (i.e. not subject to catastrophic forgetting) and "
     dataloader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
  if customOptimizer==False:
     optimizer = optim.SGD(model.parameters(), lr=LR)
     text += "standard optimizer"
  else:
     optimizer = CLASSP_optimizer(model.parameters(), lr=LR, threshold=Threshold, epsilon=Epsilon, power=Power)  #  sigma controls the decay rate of the weight update
     text += "custom optimizer"

  print("Training on combined dataset...", flush=True)
  time_sec = time.time()
  counter = 0
  for inputs, labels in dataloader:
        counter += 1
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        if customOptimizer==True:
           average_grad_sum = optimizer.average_grad_sum()
           print("average_grad_sum = " + str(average_grad_sum), flush=True)


           if counter<(4 * num_examples_dataset1/32):  #  While running the first dataset
              optimizer.step(lr=LR, threshold=Threshold, epsilon=Epsilon, power=Power, apply_scaling=False)
           else:
              optimizer.step(lr=LR, threshold=0, epsilon=Epsilon, power=Power, apply_scaling=True)
        else:
              optimizer.step()

  print("loss = " + str(loss), flush=True)
  print("Elapsed time: " + str(time.time() - time_sec) + " seconds", flush=True)

  # Calculating and printing the accuracy on both datasets
  test_dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=True)
  test_dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=True)
  accuracy1 = calculate_accuracy(model, test_dataloader1)
  accuracy2 = calculate_accuracy(model, test_dataloader2)
  print('Accuracy on dataset#1 ' + text + ': ' + str(accuracy1*100) + '%')
  print('Accuracy on dataset#2 ' + text + ': ' + str(accuracy2*100) + '%')
  return(accuracy1*100, accuracy2*100)

Acc1=[]
Acc2=[]

for n in range(n_experiments):
  print("Training sequentially with the custom optimizer iteration=" + str(n) + "...", flush=True)
  criterion = nn.CrossEntropyLoss()
  model = MyModel()
  acc1, acc2 = train(True, False)
  Acc1.append(acc1)
  Acc2.append(acc2)
print(Acc1)
print(Acc2)
