# Firewire

> Neurons that fire together, wire together.

mnist genetic algorithm

Example usage:

```bash
python mnist.py --mutation-rate 0.01 --perturbation-rate 0.1 --perturbation-strength 0.1 --population 100
```

Base MNIST example from [official PyTorch examples](https://github.com/pytorch/examples/blob/main/mnist/main.py).

## TODO

it should be monotonic improving since best models are included in the next generation. seems to be solved by not using deepcopy.