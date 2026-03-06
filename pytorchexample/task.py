"""pytorchexample: A Flower / PyTorch app."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import pandas as pd
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self, vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = 8,
                 hidden_dim: int = 256,
                 num_layers: int = 2):
        super(Net, self).__init__()

        #Embedding
        #maps each character index → dense vector of size 8.
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        #LSTM
        self.lstm == nn.LSTM(
            input_size = embed_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
        )

        #Linear classifierß
        self.fc = nn.Linear(hidden_dim, vocab_size) #num_classes number of unique players

    def forward(self, x):
        # x shape: (batch_size, seq_len) 

        embedded = self.embedding(x)

        lstm_out, _ = self.lstm(embedded)

        logits = self.fc(lstm_out)
        return logits

partitioner = None  # Cache partitioner

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #converts PIL images to tensors (text data)


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global partitioner
    if partitioner is None:
        df = pd.read_csv('./data/Shakespeare_cleaned.csv')
        hf_dataset = Dataset.from_pandas(df) #load into csv into pandas dataframe
        # make partitioner 
        partitioner = NaturalIdPartitioner(partition_by="Player")
        partitioner.dataset = hf_dataset
    
    # load partition 
    partition = partitioner.load_partition(partition_id)
    
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=67)
    
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
 

    
