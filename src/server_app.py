"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    
    c: float = context.run_config["c"]
    b: int = context.run_config["b"]
    
    # define global evaluate within main to pass in b
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""
        # Load the model and initialize it with the received weights
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load entire test set
        test_dataloader = load_centralized_dataset(batch_size=b)

        # Evaluate the global model on the test set
        test_loss, test_acc = test(model, test_dataloader, device)

        # Return the evaluation metrics
        print(f"[Round {server_round}/{num_rounds}] loss={test_loss:.4f}  accuracy={test_acc:.4f}")
        return MetricRecord({"accuracy": test_acc, "loss": test_loss})

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy with hyperparameter 'c'
    strategy = FedAvg(fraction_train=c)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # uncomment to save final model to disk if needed
    '''print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")'''
