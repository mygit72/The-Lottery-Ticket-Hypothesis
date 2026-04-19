import torch
import torch.optim as optim
import os
import json
import matplotlib.pyplot as plt
from models.lenet import LeNet300
from pruning.imp import LotteryTicketPruner
from datasets.loaders import get_mnist_loaders
from utils.trainer import Trainer

def run_lottery_ticket_experiment(rounds=5, epochs_per_round=10, lr=1.2e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model and data
    model = LeNet300().to(device)
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=60)
    pruner = LotteryTicketPruner(model, pruning_rate_fc=0.2)
    
    results = []
    
    for r in range(rounds):
        sparsity = pruner.get_sparsity()
        print(f"\n--- Round {r} (Sparsity: {sparsity:.2%}) ---")
        
        # Reset weights to initial values (except for the first round)
        if r > 0:
            pruner.reset_weights()
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        trainer = Trainer(model, train_loader, val_loader, test_loader, device)
        
        # Train
        history = trainer.train(optimizer, epochs_per_round, pruner)
        
        # Evaluate on test set
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"Test Accuracy: {test_acc:.4f}, Early Stop Iter: {history['early_stop_iter']}")
        
        results.append({
            'round': r,
            'sparsity': sparsity,
            'test_acc': test_acc,
            'early_stop_iter': history['early_stop_iter'],
            'history': history
        })
        
        # Prune for the next round
        if r < rounds - 1:
            pruner.prune_step()
            
    return results

def plot_results(results):
    sparsities = [1 - r['sparsity'] for r in results] # Percent of weights remaining
    accuracies = [r['test_acc'] for r in results]
    iters = [r['early_stop_iter'] for r in results]
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:blue'
    ax1.set_xlabel('Percent of Weights Remaining')
    ax1.set_ylabel('Test Accuracy', color=color)
    ax1.plot(sparsities, accuracies, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.invert_xaxis()
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Early Stop Iteration', color=color)
    ax2.plot(sparsities, iters, marker='s', linestyle='--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Lottery Ticket Hypothesis: LeNet-300-100 on MNIST')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plt.savefig(os.path.join(base_dir, 'experiments', 'plots', 'lottery_ticket_results.png'))
    plt.show()

if __name__ == "__main__":
    os.makedirs('experiments/plots', exist_ok=True)
    results = run_lottery_ticket_experiment(rounds=10, epochs_per_round=5)
    
    # Ensure directories exist relative to the script location or use absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = os.path.join(base_dir, 'experiments', 'runs')
    plots_dir = os.path.join(base_dir, 'experiments', 'plots')
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    with open(os.path.join(runs_dir, 'results.json'), 'w') as f:
        json.dump(results, f)
        
    plot_results(results)
