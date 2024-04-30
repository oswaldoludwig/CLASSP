# Training the model on dataset2 using EWC after having trained it on dataset1
# Author: Oswaldo Ludwig

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn, optim
import copy

epochs = 1
PATH = "./model_and_optimizer_state"

# Define your model
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

# Define a function to calculate accuracy
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

# Define a function to compute the EWC loss
def ewc_loss_eye_FIM(model, old_model, lambda_ewc):
    loss = 0
    for (name, param), (_, old_param) in zip(model.named_parameters(), old_model.named_parameters()):
        loss += torch.sum(param * param - 2 * param * old_param + old_param * old_param)
    return lambda_ewc * loss


def ewc_loss(model, old_model, Fisher_matrix, lambda_ewc):
    loss = 0
    for (name, param), (_, old_param), (_, fisher) in zip(model.named_parameters(), old_model.named_parameters(), Fisher_matrix.items()):
        loss += torch.sum(fisher * (param - old_param) ** 2)
    return lambda_ewc * loss


print("Loading the datasets...", flush=True)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
print("MNIST loaded...", flush=True)
dataset2 = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
print("FashionMNIST loaded...", flush=True)


# Initialize the model and optimizer
model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=0.25)
criterion = nn.CrossEntropyLoss()

# Initialize the Fisher Information Matrix as a dictionary
fisher_matrix = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

# Train on the first dataset just one epoch to get the Fisher information matrix
print("Training one epoch on the first dataset to calculate the FIM ...", flush=True)
dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=True)
iterations_first_dataset = 0
for inputs, labels in dataloader1:
        iterations_first_dataset += 1
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Compute the outer product of the gradients and add to the Fisher Information Matrix
        for name, param in model.named_parameters():
            grad = param.grad.detach()
            fisher_matrix[name] += grad * grad

print("loss = " + str(loss), flush=True)

# Average the Fisher Information Matrix over the number of data samples
num_samples = len(dataset1)
for name in fisher_matrix:
    fisher_matrix[name] /= num_samples

# Save the model parameters after training on the first dataset
old_model = copy.deepcopy(model)

combined_dataset = ConcatDataset([dataset1, dataset1, dataset1, dataset1, dataset2])


# Initialize the model and optimizer
model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=0.25)
criterion = nn.CrossEntropyLoss()

# Train on the both datasets using EWC
print("Training on the both dataset in sequence (no shuffling) using EWC...", flush=True)
dataloader2 = DataLoader(combined_dataset, batch_size=32, shuffle=False)
for epoch in range(epochs):  # number of epochs
    iteration = 0
    for inputs, labels in dataloader2:
        iteration += 1
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if iteration > iterations_first_dataset:  # no EWC for the first dataset (this operation wasn't possible for TF1 with static graph, but it's okay for Pytorch, which uses dynamic graph
           loss += ewc_loss(model, old_model, fisher_matrix, lambda_ewc=5000)  # Add the EWC loss
           #print("Adding EWC loss", flush=True)
        loss.backward()
        optimizer.step()
    print("loss = " + str(loss), flush=True)

# Calculate and print the accuracy on both datasets
test_dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=True)
test_dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=True)
accuracy1 = calculate_accuracy(model, test_dataloader1)
accuracy2 = calculate_accuracy(model, test_dataloader2)
print('Accuracy on dataset1: ' + str(accuracy1*100) + '%')
print('Accuracy on dataset2: ' + str(accuracy2*100) + '%')
