# Neural Networks: Zero to Hero

Code and Jupyter notebook implementations following Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) playlist.

## Structure

- **L1_micrograd/**
  - `enginep1.ipynb`: Custom scalar autograd engine and backpropagation from scratch.
  - `enginep2.ipynb`: Multi-layer perceptron (Neuron, Layer, MLP) built on the autograd engine.
- **L2_makemore/**
  - `Indian_Names.txt`: Dataset of ~54,000 Indian names for autoregressive language modeling.
  - `engine_bigram.ipynb`: Bigram character-level language model (count matrix and single-layer neural net).
  - `engine_mlp.ipynb`: MLP character-level language model based on Bengio et al. (2003).

## Requirements

```bash
pip install torch numpy matplotlib graphviz jupyter
```

## Usage

Start Jupyter notebook and open any notebook to run:

```bash
jupyter notebook
```

## Reference

- YouTube Playlist: [Andrej Karpathy - Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
