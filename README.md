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

## Status

* It learns, strictly monotonic fitness improvement - on the train set!
* Optimizing directly for accuracy, not for loss