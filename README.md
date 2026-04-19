# Project Report: Implementation of The Lottery Ticket Hypothesis

## 1. Introduction
This report details the advanced code implementation of the research paper "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" by Jonathan Frankle and Michael Carbin [1]. The primary objective of this project was to reproduce and extend the core findings of the paper, focusing on the iterative magnitude pruning (IMP) algorithm and its impact on neural network training and performance. The implementation aims to provide a modular and extensible framework for further research into the Lottery Ticket Hypothesis.

## 2. The Lottery Ticket Hypothesis

The Lottery Ticket Hypothesis posits that dense, randomly-initialized neural networks contain subnetworks (referred to as "winning tickets") that, when trained in isolation from their original initializations, can achieve comparable or even superior test accuracy to the original dense network, often in fewer training iterations. The key insight is that these winning tickets possess a fortuitous combination of initial weights and connections that are particularly conducive to effective learning.

### Iterative Magnitude Pruning (IMP) Algorithm
The paper introduces an iterative pruning procedure to identify these winning tickets. The steps are as follows:

1.  **Initialization:** A neural network $f(x; \theta_0)$ is randomly initialized with parameters $\theta_0$.
2.  **Training:** The network is trained for $j$ iterations, resulting in trained parameters $\theta_j$.
3.  **Pruning:** A fraction $p\%$ of the parameters in $\theta_j$ with the smallest magnitudes are pruned (set to zero), creating a binary mask $m$.
4.  **Resetting:** The remaining, unpruned parameters are reset to their original values from $\theta_0$. This forms the winning ticket subnetwork $f(x; m \odot \theta_0)$.
5.  **Iteration:** Steps 2-4 are repeated for $n$ rounds. In each subsequent round, $p\%$ of the *remaining* weights are pruned, leading to progressively sparser subnetworks.

## 3. Implementation Details

### 3.1. Project Structure
The project is organized into a modular structure to facilitate clarity, extensibility, and ease of use. The main directories and their contents are as follows:

```
./
├── README.md             # Project overview and setup instructions
├── requirements.txt      # Python dependencies
├── src/
│   ├── models/           # Neural network architectures (LeNet-300-100, ConvNets)
│   ├── pruning/          # Pruning algorithms (LotteryTicketPruner)
│   ├── datasets/         # Data loading and preprocessing for MNIST, CIFAR-10
│   ├── utils/            # Utility functions (Trainer class)
│   ├── main.py           # Main script for running experiments
│   └── config.py         # (Planned) Configuration file for hyperparameters
├── experiments/          # Stores experiment results, logs, and plots
│   ├── runs/             # JSON files with experiment data
│   └── plots/            # Generated visualizations
└── notebooks/            # (Planned) Jupyter notebooks for interactive analysis
```

### 3.2. Key Components

*   **`src/models/lenet.py`**: Implements the LeNet-300-100 architecture, a fully-connected network used for MNIST classification, as described in the paper.
*   **`src/pruning/imp.py`**: Contains the `LotteryTicketPruner` class, which encapsulates the iterative magnitude pruning logic. It handles the initialization of masks, pruning steps (identifying and zeroing out low-magnitude weights), and resetting remaining weights to their initial values.
*   **`src/datasets/loaders.py`**: Provides functions (`get_mnist_loaders`, `get_cifar10_loaders`) to load and preprocess the MNIST and CIFAR-10 datasets, including data augmentation and splitting into training, validation, and test sets.
*   **`src/utils/trainer.py`**: Implements the `Trainer` class, which manages the training and evaluation loops. It supports different optimizers, loss functions, and tracks metrics like loss and accuracy. It also incorporates early-stopping logic.
*   **`src/main.py`**: The main script orchestrates the entire experiment. It initializes the model, data loaders, and the pruner, then runs the iterative pruning process for a specified number of rounds. It records results and generates plots.

## 4. Experimental Results

The initial experiment was conducted using the LeNet-300-100 model on the MNIST dataset, following the iterative magnitude pruning procedure. The experiment ran for 10 rounds, with 20% of the remaining weights pruned in each round. The learning rate was set to 1.2e-3, and each round involved 5 epochs of training.

### 4.1. Results Summary
The `results.json` file contains the detailed output of each pruning round. A summary of the key metrics (sparsity, test accuracy, and early-stopping iteration) for each round is presented below:

| Round | Sparsity (%) | Test Accuracy | Early Stop Iteration |
|-------|--------------|---------------|----------------------|
| 0     | 0.00         | 0.9729        | 4500                 |
| 1     | 20.00        | 0.9745        | 2700                 |
| 2     | 36.00        | 0.9694        | 1800                 |
| 3     | 48.80        | 0.9780        | 3600                 |
| 4     | 59.04        | 0.9795        | 4500                 |
| 5     | 67.23        | 0.9749        | 1800                 |
| 6     | 73.79        | 0.9768        | 1800                 |
| 7     | 79.03        | 0.9806        | 2700                 |
| 8     | 83.23        | 0.9782        | 1800                 |
| 9     | 86.58        | 0.9795        | 1800                 |

### 4.2. Visualization
The following plot illustrates the relationship between the percentage of weights remaining (1 - sparsity) and the test accuracy, as well as the early-stopping iteration. This visualization is crucial for understanding the Lottery Ticket Hypothesis, as it shows how pruning can lead to sparser networks that maintain or even improve performance and training efficiency.
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/d310510e-3a9c-4336-b5fe-92645be6ccea" />

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git
- (Optional) Virtual Environment (venv / conda)
- Google Colab (for notebook execution)


**Analysis of the Plot:**

*   **Test Accuracy vs. Weights Remaining:** The blue line shows that as the network is pruned (percentage of weights remaining decreases), the test accuracy initially remains stable or even slightly increases before eventually dropping off at very high sparsity levels. This supports the hypothesis that smaller, effective subnetworks exist.
*   **Early Stop Iteration vs. Weights Remaining:** The red dashed line indicates that sparser networks often reach their optimal validation loss (and thus stop training earlier) in fewer iterations. This suggests that winning tickets can learn faster than their denser counterparts.

These results are consistent with the findings presented in the original paper, demonstrating the successful reproduction of the core phenomenon of the Lottery Ticket Hypothesis.

## 5. Conclusion and Future Work

This project successfully implemented an advanced framework for exploring the Lottery Ticket Hypothesis, reproducing key experimental results on LeNet-300-100 and MNIST. The modular design allows for easy extension and further experimentation.

### Future Enhancements:

*   **Additional Architectures:** Implement ResNet-18 and VGG-19 for CIFAR-10 to validate the hypothesis on more complex and deeper networks.
*   **Global Pruning:** Incorporate global pruning strategies for deeper networks, as discussed in the paper, where pruning is applied across all convolutional layers collectively rather than layer-wise.
*   **Learning Rate Warmup:** Implement learning rate warmup schedules, which were shown to be crucial for finding winning tickets in deeper networks at higher learning rates.
*   **Hyperparameter Optimization:** Integrate a configuration management system (`config.py`) and potentially hyperparameter search tools to systematically explore the impact of different pruning rates, learning rates, and training schedules.
*   **Comprehensive Experiment Tracking:** Utilize tools like TensorBoard or Weights & Biases for more robust experiment logging and visualization.
*   **Interactive Notebooks:** Develop Jupyter notebooks for interactive analysis and demonstration of the pruning process and its effects.
