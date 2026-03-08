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
    ''' LSTM model
    8-dimensional character embedding, 
    2 LSTM layers with 256 hidden units,
    vocabulary of 95 unique characters (printable ASCII range).
    note: paper only has 80, but this might not be the same dataset
    '''
    def __init__(self):
        super(Net, self).__init__()

        # vocab of 95 letters into 8-dim vector
        self.embedding = nn.Embedding(95, 8)
        
        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(256, 95)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len) of character indices
        embeds = self.embedding(x)            # (batch, seq_len, embed_dim)
        out, hidden = self.lstm(embeds, hidden)  # out: (batch, seq_len, hidden_dim)
        logits = self.fc(out)                 # (batch, seq_len, vocab_size)
        return logits, hidden


partitioner = None  # Cache partitioner
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# TODO
def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data."""
    # only initialize partitioner once
    global partitioner
    if partitioner is None:
        df = pd.read_csv('./data/Shakespeare_cleaned.csv')
        hf_dataset = Dataset.from_pandas(df)
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

# TODO
def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    df = pd.read_csv('./data/Shakepseare_cleaned.csv')
    dataset = Dataset.from_pandas(df)
    # TODO: dataset transformations
    return DataLoader(dataset, batch_size=128)

# TODO
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

# TODO
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
 

    
