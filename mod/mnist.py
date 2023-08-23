from __future__ import print_function
import argparse
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.activations = {}
        def get_activation(name):
            def hook(layer, input, output):
                # TODO deactivate this during "training", only used in genetic algorithm
                self.activations[name] = output.detach()
            return hook
        self.conv1.register_forward_hook(get_activation('conv1'))
        self.conv2.register_forward_hook(get_activation('conv2'))
        self.fc1.register_forward_hook(get_activation('fc1'))
        self.fc2.register_forward_hook(get_activation('fc2'))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, models, device, train_loader, optimizers, epoch):
    for model in models:
        model.train()
        model.store_activations(True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for model, optimizer in zip(models, optimizers):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Regular Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def train_genetic(args, models, device, train_loader, optimizers, epoch):
    print(f"Num models: {len(models)}") 

    for model in models:
        model.train()
    losses = {x: 0. for x in range(len(models))}
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for i, model in enumerate(models):
            if args.use_dropout:
                model.train()
            else:
                model.eval()
            with torch.no_grad():
                output = model(data)
                # optimize for num correct, directly
                # loss = F.nll_loss(output, target)
                # losses[i] += loss.item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                losses[i] += pred.eq(target.view_as(pred)).sum().item()
            # loss.backward()
            # optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Genetic Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader)))
    # get top 2 models
    best = [k for k, v in sorted(losses.items(), key=lambda item: item[1], reverse=True)]
    scores = [losses[x] for x in best]
    
    # clone top 2 models
    # TODO if best1 is highly correlated with best0, choose a different best1
    best_0 = deepcopy(models[best[0]])
    best_1 = deepcopy(models[best[1]])

    # batch_idx, (data, target) = next(enumerate(train_loader))
    # data, target = data.to(device), target.to(device)
    # best_0(data)
    # best_1(data)
    # don't actually need to run a pass, just take the last pass
    covariance_matrix = torch.cov(torch.cat((best_0.activations['fc1'], best_1.activations['fc1'])))
    # TODO


    new_models = []
    if not args.growing_population:
        models = models[:-2]
    for model in models:
        new_models.append(Net())
        for param1, param2, existing, new in zip(best_0.parameters(), best_1.parameters(), model.parameters(), new_models[-1].parameters()):
            # TODO don't train dropped-out params
            # import ipdb; ipdb.set_trace()
            # crossover
            new.data = param1.data.clone() if torch.rand(1) > 0.5 else param2.data.clone()
            # peturbation
            if torch.rand(1) < args.perturbation_rate:
                new.data += torch.clamp(torch.randn_like(existing.data) * 0.5, -1, 1) * args.perturbation_strength
            # mutation
            if torch.rand(1) < args.mutation_rate:
                new.data = torch.clamp(torch.randn_like(existing.data) * 0.5, -1, 1)
        new_models[-1].to(device).train()
    
    # add top 2 models to population
    # results in a growing population
    best_0 = best_0.to(device).train()
    best_1 = best_1.to(device).train()

    new_models += [best_1, best_0]

    return new_models


def test(model, device, test_loader, train_loader):
    losses = {"test_loss": 0., "correct": 0.}
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses["test_loss"] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            losses["correct"] += pred.eq(target.view_as(pred)).sum().item()

    losses["test_loss"] /= len(test_loader.dataset)
    
    print('\nTest set: Best loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        losses["test_loss"], losses["correct"], len(test_loader.dataset),
        100. * losses["correct"] / len(test_loader.dataset)))

    losses = {"test_loss": 0., "correct": 0.}
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses["test_loss"] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            losses["correct"] += pred.eq(target.view_as(pred)).sum().item()

        losses["test_loss"] /= len(test_loader.dataset)
    
    print('\nTrain set: Best loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        losses["test_loss"], losses["correct"], len(train_loader.dataset),
        100. * losses["correct"] / len(train_loader.dataset)))
    
    model.train()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--mutation-rate', type=float, default=0.01, metavar='MT',
                        help='Mutation rate (default: 0.01)')
    parser.add_argument('--perturbation-rate', type=float, default=0.01, metavar='PR',
                        help='Perturbation rate (default: 0.01)')
    parser.add_argument('--perturbation-strength', type=float, default=0.01, metavar='PS',
                        help='Perturbation strength (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--backprop', action='store_true', default=False,
                        help='Do an epoch of backprop before starting genetic algorithm')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--population', type=int, default=10, metavar='POP',
                        help='population size (default: 10)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--growing-population', action='store_true', default=False,
                        help='If set, population will grow by 2 each generation (best 2 models are added to population)')
    parser.add_argument('--use-dropout', action='store_true', default=False,
                        help='If set, dropout will be active (model.train) and evolution not monotonic')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    models = [Net().to(device) for _ in range(args.population)]
    optimizers = [optim.Adadelta(model.parameters(), lr=args.lr) for model in models]

    schedulers = [StepLR(optimizer, step_size=1, gamma=args.gamma) for optimizer in optimizers]
    if args.backprop:
        train(args, models, device, train_loader, optimizers, 1)
    test(models[-1], device, test_loader, train_loader)
    for epoch in range(1, args.epochs + 1):
        models = train_genetic(args, models, device, train_loader, optimizers, epoch)
        test(models[-1], device, test_loader, train_loader) # last model of next generation is best
        for scheduler in schedulers:
            scheduler.step()

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()