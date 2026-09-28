# Neural Networks: Zero to Hero

Code and Jupyter notebook implementations following Andrej Karpathy's acclaimed [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) playlist.

This repository tracks hands-on implementations building deep learning systems from the ground up—starting with scalar autograd engines and fundamental neural network mechanics, progressing to autoregressive character-level language models with MLPs, manual backprop verification, Batch Normalization, and hierarchical WaveNets.

---

## Repository Structure

```text
.
├── L1_micrograd/
│   ├── enginep1.ipynb               # Scalar autograd engine & backward pass from scratch
│   └── enginep2.ipynb               # Multi-Layer Perceptron (Neuron, Layer, MLP) on Micrograd
│
├── L2_makemore/
│   ├── Indian_Names.txt             # Dataset (~54k Indian names for autoregressive language modeling)
│   ├── MLP.pdf                      # Bengio et al. (2003) neural language model reference
│   ├── engine_bigram.ipynb          # Part 1: Bigram character-level LM (Counts vs. Neural Net)
│   ├── engine_mlp1.ipynb            # Part 2: MLP character-level LM (Bengio et al. 2003)
│   ├── engine_mlp2.ipynb            # Part 3: Activations, Gradients & Batch Normalization
│   ├── becoming_a_backprop_ninja.ipynb # Part 4: Manual tensor-level backpropagation
│   └── engine_webnets.ipynb         # Part 5: WaveNet architecture with hierarchical flattening
│
└── README.md
```

---

## Detailed Module Breakdown

### [L1_micrograd](file:///e:/Andrej_Karpathy_playlist/L1_micrograd)
Building `micrograd`—a tiny scalar-valued autograd engine with a PyTorch-like API and a neural net library built on top of it.

- **[enginep1.ipynb](file:///e:/Andrej_Karpathy_playlist/L1_micrograd/enginep1.ipynb)**:
  - Custom `Value` scalar class with forward operations (`+`, `*`, `pow`, `tanh`, `exp`, etc.).
  - Reverse-mode automatic differentiation (backpropagation) via topological sorting of the computation graph.
  - Computation graph visualization with `graphviz`.
- **[enginep2.ipynb](file:///e:/Andrej_Karpathy_playlist/L1_micrograd/enginep2.ipynb)**:
  - Building `Neuron`, `Layer`, and `MLP` classes from raw `Value` scalars.
  - Training an MLP on toy classification datasets with loss computation, forward pass, zeroing gradients, and gradient descent updates.

---

### [L2_makemore](file:///e:/Andrej_Karpathy_playlist/L2_makemore)
Building `makemore`—an autoregressive character-level language model trained on a custom dataset of ~54,000 Indian names (`Indian_Names.txt`).

- **[engine_bigram.ipynb](file:///e:/Andrej_Karpathy_playlist/L2_makemore/engine_bigram.ipynb)** — *Building makemore Part 1: Bigrams*
  - Bigram character-level language modeling using a 2D count matrix and normalization.
  - Negative Log Likelihood (NLL) loss formulation and model evaluation.
  - Equivalent formulation using a single-layer neural network trained via PyTorch tensors, one-hot encoding, and gradient descent.

- **[engine_mlp1.ipynb](file:///e:/Andrej_Karpathy_playlist/L2_makemore/engine_mlp1.ipynb)** — *Building makemore Part 2: MLP (Bengio et al. 2003)*
  - Embedding characters into a continuous lookup table (vector space).
  - Multi-layer perceptron architecture with context windows, hidden layers (`tanh`), and softmax outputs.
  - Cross-entropy loss computation, mini-batch gradient descent, learning rate decay schedules, and train/val/test splits.

- **[engine_mlp2.ipynb](file:///e:/Andrej_Karpathy_playlist/L2_makemore/engine_mlp2.ipynb)** — *Building makemore Part 3: Activations & Batch Normalization*
  - Diagnosing network initialization issues: saturated activations, dead neurons, and vanishing/exploding gradients.
  - Kaiming initialization / weight scaling for stable `tanh` activations.
  - Implementing `BatchNorm1d` from scratch (forward pass, batch mean/variance, running statistics, scale `gamma` & shift `beta`).
  - PyTorch-ifying layers into clean modular classes (`Linear`, `BatchNorm1d`, `Tanh`).

- **[becoming_a_backprop_ninja.ipynb](file:///e:/Andrej_Karpathy_playlist/L2_makemore/becoming_a_backprop_ninja.ipynb)** — *Building makemore Part 4: Backprop Ninja*
  - Manual derivation and implementation of analytical gradients (backpropagation) for every intermediate tensor:
    - Cross-entropy loss & softmax derivatives
    - Batch Normalization layer derivatives (exact simplified expressions)
    - Matrix multiplications, broadcasting, and slice/indexing gradients
  - Exact gradient verification and numerical checking against PyTorch `backward()`.

- **[engine_webnets.ipynb](file:///e:/Andrej_Karpathy_playlist/L2_makemore/engine_webnets.ipynb)** — *Building makemore Part 5: WaveNet Architecture*
  - Implementing the hierarchical tree architecture inspired by DeepMind's WaveNet (van den Oord et al., 2016).
  - Receptive field expansion by progressive consecutive character aggregation using custom `FlattenConsecutive` layer.
  - Deep character-level neural network pipeline using `Sequential` container with `Embedding`, `Linear`, and `BatchNorm1d`.

---

## Requirements & Installation

Make sure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch numpy matplotlib graphviz jupyter
```

*Note: For graph visualization in `L1_micrograd`, ensure `graphviz` is installed on your system (`winget install Graphviz` on Windows, or `brew install graphviz` on macOS).*

---

## Usage

Launch Jupyter Notebook from the root of the repository:

```bash
jupyter notebook
```

Navigate to either `L1_micrograd/` or `L2_makemore/` and run the notebooks interactively.

---

## References & Resources

- **Course Playlist**: [Andrej Karpathy - Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- **Micrograd Repository**: [karpathy/micrograd](https://github.com/karpathy/micrograd)
- **Makemore Repository**: [karpathy/makemore](https://github.com/karpathy/makemore)
- **Bengio et al. (2003)**: *A Neural Probabilistic Language Model*
- **van den Oord et al. (2016)**: *WaveNet: A Generative Model for Raw Audio*
