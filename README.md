# Firewire

> Neurons that fire together, wire together.

mnist genetic algorithm

Example usage:

```bash
python mnist.py --mutation-rate 0.01 --perturbation-rate 0.1 --perturbation-strength 0.1 --population 100
```

Base MNIST example from [official PyTorch examples](https://github.com/pytorch/examples/blob/main/mnist/main.py).

## TODO

* how to tackle dropout in a genetic algo? It was preventing monotonic fitness improvement
* eliminating dropout makes it overfit quite bad
* actually implement firewire

# NOTES

* Activations of layers of best 2 models don't seem to correlate - each network is learning different things. (TODO a backprop training step first, that probably makes it more homogeneous. Done, and true, but that's probably because there *is* an optimal answer so you'd expect them to converge on it.)
* conv layers are much more correlated than fc layers

## Status

* It learns, strictly monotonic fitness improvement - on the train set!
* Optimizing directly for accuracy, not for loss