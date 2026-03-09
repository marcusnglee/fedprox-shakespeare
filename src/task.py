"""pytorchexample: A Flower / PyTorch app."""
import torch
import torch.nn as nn
from datasets import Dataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from torch.utils.data import DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

partitioner = None  # Cache partitioner

class Net(nn.Module):
    ''' LSTM model
    8-dimensional character embedding, 
    2 LSTM layers with 256 hidden units,
    vocabulary of 96 unique characters (printable ASCII range + padding).
    note: paper only has 80, but this might not be the same dataset
    '''
    def __init__(self):
        super(Net, self).__init__()

        # vocab of 96 letters into 8-dim vector
        self.embedding = nn.Embedding(96, 8)
        
        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(256, 96)

    def forward(self, x, hidden=None):
        
        embeds = self.embedding(x)         
        out, _ = self.lstm(embeds, hidden) 
        logits = self.fc(out)           
        return logits

'''
transform the player lines into input and targets char arrays (str)
'''
def apply_transforms(batch):
    encoded = [encode(line) for line in batch['PlayerLine'] ]
    batch['input_ids'] = [torch.tensor(line[:-1], dtype=torch.long) for line in encoded]
    batch['target_ids'] = [torch.tensor(line[1:], dtype=torch.long) for line in encoded]
    return batch

# since lines are not of similar length, pad them with zeroes
def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    target_ids = pad_sequence([item['target_ids'] for item in batch], batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "target_ids": target_ids}

'''encodes lines into integer list of range (1, 95)'''
def encode(text: str) -> list[int]:
    # minus 31 because ascii starts at ord 32
    # 0 will be our padding value
    return [ord(c) - 31 for c in text if 0 < ord(c) - 31 < 96] # drop unknown chars


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition data."""
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
        partition_train_test["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size, collate_fn=collate_fn)
    return trainloader, testloader

def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire dataset as test set
    df = pd.read_csv('./data/Shakespeare_cleaned.csv')
    dataset = Dataset.from_pandas(df)
    dataset = dataset.with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128, collate_fn=collate_fn)

"""Train the model on the training set."""
def train(net, trainloader, epochs, lr, device):
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            optimizer.zero_grad()
            # must permute from shape (batch, seq_len, 96) -> (batch, 96, seq_len)
            loss = criterion(net(input_ids).permute(0, 2, 1), target_ids)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss

# TODO
def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            outputs = net(input_ids)
            loss += criterion(outputs.permute(0, 2, 1), target_ids).item()
            # TODO: understand this better
            preds = torch.max(outputs, dim=2)[1]         # argmax over vocab → (batch, seq_len)
            mask = target_ids != 0
            correct += (preds[mask] == target_ids[mask]).sum().item()
            total_tokens += mask.sum().item()
    accuracy = correct / total_tokens # divide by number of chars, not lines
    loss = loss / len(testloader) # avg loss across batch means 
    return loss, accuracy