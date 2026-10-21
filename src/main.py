import argparse
import copy
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from datasets.loaders import get_mnist_loaders
from models.lenet import LeNet300
from pruning.imp import LotteryTicketPruner
from utils.seed import set_seed
from utils.trainer import Trainer

# Okabe-Ito colorblind-safe palette: one fixed color per series, used consistently
# across every plot in this project.
COLOR_WINNING_TICKET = "#0072B2"  # blue
COLOR_RANDOM_REINIT = "#D55E00"   # vermillion


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterative Magnitude Pruning (Lottery Ticket Hypothesis) on LeNet-300-100 / MNIST"
    )
    parser.add_argument("--rounds", type=int, default=10, help="Number of pruning rounds")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per round")
    parser.add_argument("--lr", type=float, default=1.2e-3, help="Adam learning rate")
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--pruning-rate-fc", type=float, default=0.2, help="Fraction of remaining FC weights pruned per round")
    parser.add_argument("--pruning-rate-conv", type=float, default=0.1, help="Fraction of remaining conv weights pruned per round")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-control", action="store_true", help="Skip the random-reinitialization control experiment")
    parser.add_argument("--quiet", action="store_true", help="Disable per-batch progress bars")
    parser.add_argument("--output-dir", type=str, default="experiments", help="Where to write results/plots")
    return parser.parse_args()


def run_lottery_ticket_experiment(rounds=10, epochs_per_round=5, lr=1.2e-3, batch_size=60,
                                   pruning_rate_fc=0.2, pruning_rate_conv=0.1, seed=42,
                                   run_control=True, verbose=True):
    """
    Runs iterative magnitude pruning and, for each resulting sparsity level, a
    random-reinitialization control: the same mask applied to a freshly
    initialized network instead of the winning ticket's original weights
    theta_0. Comparing the two is the actual test of the Lottery Ticket
    Hypothesis -- that the *initialization*, not just the sparse topology,
    is what makes the winning ticket trainable.
    """
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = LeNet300().to(device)
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=batch_size, seed=seed)
    pruner = LotteryTicketPruner(model, pruning_rate_fc=pruning_rate_fc, pruning_rate_conv=pruning_rate_conv)

    results = []

    for r in range(rounds):
        sparsity = pruner.get_sparsity()
        print(f"\n--- Round {r} (Sparsity: {sparsity:.2%}) ---")

        # Reset weights to initial values theta_0 (except for the first round)
        if r > 0:
            pruner.reset_weights()

        optimizer = optim.Adam(model.parameters(), lr=lr)
        trainer = Trainer(model, train_loader, val_loader, test_loader, device)
        history = trainer.train(optimizer, epochs_per_round, masks=pruner.masks, verbose=verbose)
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"Winning ticket   -> Test Accuracy: {test_acc:.4f}, Early Stop Iter: {history['early_stop_iter']}")

        round_result = {
            'round': r,
            'sparsity': sparsity,
            'test_acc': test_acc,
            'early_stop_iter': history['early_stop_iter'],
            'history': history,
        }

        # Random-reinitialization control: same mask, fresh random weights.
        if run_control:
            control_model = LeNet300().to(device)
            pruner.apply_mask_to_model(control_model)
            control_optimizer = optim.Adam(control_model.parameters(), lr=lr)
            control_trainer = Trainer(control_model, train_loader, val_loader, test_loader, device)
            control_history = control_trainer.train(control_optimizer, epochs_per_round, masks=pruner.masks, verbose=verbose)
            control_test_loss, control_test_acc = control_trainer.evaluate(test_loader)
            print(f"Random reinit    -> Test Accuracy: {control_test_acc:.4f}, Early Stop Iter: {control_history['early_stop_iter']}")

            round_result['control_test_acc'] = control_test_acc
            round_result['control_early_stop_iter'] = control_history['early_stop_iter']
            round_result['control_history'] = control_history

        results.append(round_result)

        # Prune for the next round
        if r < rounds - 1:
            pruner.prune_step()

    return results


def plot_results(results, output_dir):
    weights_remaining = [1 - r['sparsity'] for r in results]  # fraction of weights remaining
    accuracies = [r['test_acc'] for r in results]
    iters = [r['early_stop_iter'] for r in results]
    has_control = 'control_test_acc' in results[0]

    fig, (ax_acc, ax_iter) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_acc.plot(weights_remaining, accuracies, marker='o', color=COLOR_WINNING_TICKET, label='Winning ticket')
    ax_acc.set_xlabel('Weights remaining')
    ax_acc.set_ylabel('Test accuracy')
    ax_acc.set_xscale('log')
    ax_acc.invert_xaxis()
    ax_acc.set_title('Test accuracy vs. sparsity')
    ax_acc.grid(True, which='both', alpha=0.3)

    ax_iter.plot(weights_remaining, iters, marker='o', color=COLOR_WINNING_TICKET, label='Winning ticket')
    ax_iter.set_xlabel('Weights remaining')
    ax_iter.set_ylabel('Early-stop iteration')
    ax_iter.set_xscale('log')
    ax_iter.invert_xaxis()
    ax_iter.set_title('Training speed vs. sparsity')
    ax_iter.grid(True, which='both', alpha=0.3)

    if has_control:
        control_accuracies = [r['control_test_acc'] for r in results]
        control_iters = [r['control_early_stop_iter'] for r in results]
        ax_acc.plot(weights_remaining, control_accuracies, marker='s', linestyle='--',
                    color=COLOR_RANDOM_REINIT, label='Random reinit (control)')
        ax_iter.plot(weights_remaining, control_iters, marker='s', linestyle='--',
                     color=COLOR_RANDOM_REINIT, label='Random reinit (control)')

    ax_acc.legend()
    ax_iter.legend()
    fig.suptitle('Lottery Ticket Hypothesis: LeNet-300-100 on MNIST')
    fig.tight_layout()

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'lottery_ticket_results.png'), dpi=150)
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    results = run_lottery_ticket_experiment(
        rounds=args.rounds,
        epochs_per_round=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        pruning_rate_fc=args.pruning_rate_fc,
        pruning_rate_conv=args.pruning_rate_conv,
        seed=args.seed,
        run_control=not args.no_control,
        verbose=not args.quiet,
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, args.output_dir)
    runs_dir = os.path.join(output_dir, 'runs')
    os.makedirs(runs_dir, exist_ok=True)

    with open(os.path.join(runs_dir, 'results.json'), 'w') as f:
        json.dump({'config': vars(args), 'results': results}, f, indent=2)

    plot_results(results, output_dir)
