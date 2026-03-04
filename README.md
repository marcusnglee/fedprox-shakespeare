# FedAvg Algorithm: Shakespeare LSTM example
the goal of this project is to implement the Shakepseare dataset example from the FedAvg paper

note: initialized as a flower pytorch example 
tutorial to learn: https://github.com/adap/flower/blob/main/examples/flower-in-30-minutes/tutorial.ipynb

## implementation steps
- [x] clean data
- [ ] create partitoner to simulate federation: partition clients by 'Player' identifier
- [ ] make LSTM model architecture ( Net() inside task.py)
- [ ] implement server code (load global model, set strategy to FedAvg, define global evaluation fn)
- [ ] client.py (train function, local evaluation fn)
- [ ] run with different hyperparameters and compare