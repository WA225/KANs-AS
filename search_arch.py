import random
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, Subset

import nni.retiarii.nn.pytorch as nnr
import nni.retiarii.strategy as strategy
import nni.retiarii.evaluator.pytorch.lightning as pl
from nni.retiarii import model_wrapper, serialize
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.serializer import basic_unit

# Import your real PyKAN KANLayer
from kan.KANLayer import KANLayer  
from kan.KANLayer import extend_grid

from collections.abc import Iterable

import numpy as np

KANLayer = basic_unit(KANLayer)

def SiLU_fct(x):
    if x.dtype == torch.float16 and x.device.type == 'cpu':
        x_fp32 = x.to(torch.float32)
        return (x_fp32 / (1 + torch.exp(-x_fp32))).to(torch.float16)
    return torch.nn.functional.silu(x)  # Works for float32 and float16 on CUDA

class SiLU(nn.Module):
    def forward(self, x):
        if x.dtype == torch.float16 and x.device.type == 'cpu':
            x_fp32 = x.to(torch.float32)
            return (x_fp32 / (1 + torch.exp(-x_fp32))).to(torch.float16)
        return torch.nn.functional.silu(x)  # Works for float32 and float16 on CUDA


class ReLU(nn.Module):
    def forward(self, x):
        if x.dtype == torch.float16  and x.device.type == 'cpu':
            return torch.maximum(x, torch.tensor(0.0, dtype=torch.float16, device=x.device))
        return torch.relu(x)

class Tanh(nn.Module):
    def forward(self, x):
        if x.dtype == torch.float16  and x.device.type == 'cpu':
            x_fp32 = x.to(torch.float32)
            return torch.tanh(x_fp32).to(torch.float16)
        return torch.tanh(x)

@model_wrapper
class KANNet(nnr.Module):
    def __init__(self):
        super().__init__()
        # set_deterministic(42)
        # 5) number of hidden layers
        # self.num_hidden_layers = nnr.ValueChoice([1, 2, 3, 4], label='num_hidden_layers')
        # self.num_hidden_layers = 1
        # 2) neurons per layer (in/out dims for all hidden layers)
        # self.neurons_per_layer = nnr.ValueChoice([5], label='neurons_per_layer')
        self.neurons_per_layer = 5
        # 1) number of spline intervals
        self.num_spline_points = nnr.ValueChoice([5, 20, 35], label='num_spline_points')
        # 2) spline polynomial order
        self.spline_order = nnr.ValueChoice([1, 2, 3, 4], label='spline_order')
        # 3) smoothness (scale_sp)
        # self.smoothness = nnr.ValueChoice([1.0], label='smoothness')
        self.smoothness = 1.0
        # 4) basis function
        # self.base_fun = nnr.LayerChoice([SiLU()], label='basis_function')
        self.base_fun = SiLU_fct
        # 5) grid range
        # self.grid_range = nnr.ValueChoice([(-1.0, 1.0)], label='grid_range')
        self.grid_range = (-3, 3)
        # 6) precision
        self.precision = nnr.ValueChoice(['float16', 'float32'], label='precision')

        # Input layer
        self.input_layer = KANLayer(
            in_dim=28*28,
            out_dim=self.neurons_per_layer,  # ValueChoice
            num=self.num_spline_points,
            k=self.spline_order,
            base_fun=self.base_fun,
            grid_range=self.grid_range,
            scale_sp=self.smoothness
        )

        # Hidden stack (max 4), repeated per trial choice
        # 
        
        def builder(layer_idx):
            return KANLayer(
                    in_dim=self.neurons_per_layer,
                    out_dim=self.neurons_per_layer,  # ValueChoice
                    num=self.num_spline_points,
                    k=self.spline_order,
                    base_fun=self.base_fun,
                    grid_range=self.grid_range,
                    scale_sp=self.smoothness)

        self.hidden_layers = nn.ModuleList([
            builder(i) for i in range(4)
        ])
        # self.selected_depth = nnr.ValueChoice([3], label='hidden_depth')
        self.selected_depth = 1


        self.out_layer = KANLayer(
            in_dim=self.neurons_per_layer,  # ValueChoice
            out_dim = 10,
            num=self.num_spline_points,
            k=self.spline_order,
            base_fun=self.base_fun,
            grid_range=self.grid_range,
            scale_sp=self.smoothness
        )

    def forward(self, x):
        dtype = torch.float16 if self.precision == 'float16' else torch.float32
        x = x.view(x.size(0), -1).to(dtype=dtype)
        x = self.input_layer(x)[0]

        # stack N hidden layers
        # for layer in self.hidden_layers:
        #     x = layer(x)
        for i in range(self.selected_depth):
            x = self.hidden_layers[i](x)
            if isinstance(x, tuple):
                # print(x)
                x = x[0]
        return self.out_layer(x)[0]

def set_deterministic(seed=42):
    torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    py_rng_state = random.getstate()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch_rng_state, np_rng_state, py_rng_state

def restore_random(torch_rng_state, np_rng_state, py_rng_state):
    torch.set_rng_state(torch_rng_state)
    np.random.set_state(np_rng_state)
    random.setstate(py_rng_state)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = serialize(MNIST, root="./data/mnist", train=True,  transform=transform, download=True)
    test_ds  = serialize(MNIST, root="./data/mnist", train=False, transform=transform, download=True)
    train_loader = pl.DataLoader(train_ds, batch_size=128)
    val_loader   = pl.DataLoader(test_ds,  batch_size=128)

    # # initalize dataset. Note that 50k+10k is used. It's a little different from paper
    # transf = [
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip()
    # ]
    # normalize = [
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768])
    # ]
    # train_dataset = serialize(CIFAR10, './data/cifar10', train=True, download=True, transform=transforms.Compose(transf + normalize))
    # test_dataset = serialize(CIFAR10, './data/cifar10', train=False, transform=transforms.Compose(normalize))

    # # train_subset = Subset(train_dataset, indices=list(range(20000)))
    # # test_subset = Subset(train_dataset, indices=list(range(5000)))


    # train_loader=pl.DataLoader(train_dataset, batch_size=128, num_workers = 1)
    # val_loader=pl.DataLoader(test_dataset, batch_size=128, num_workers = 1)

    trainer = pl.Classification(
        train_dataloader=train_loader,
        val_dataloaders=val_loader,
        max_epochs=10
    )
    torch_rng_state, np_rng_state, py_rng_state = set_deterministic()
    model = KANNet()
    restore_random(torch_rng_state, np_rng_state, py_rng_state)
    strategy = strategy.Random()
    # strategy = strategy.TPEStrategy()
    exp = RetiariiExperiment(model, trainer, [], strategy)

    conf = RetiariiExeConfig('local')
    conf.experiment_name      = 'kan_mnist_search'
    conf.trial_concurrency    = 1
    conf.max_trial_number     = 25
    conf.training_service.use_active_gpu = False

    # exp.run(conf, port=8080 + random.randint(0, 100))
    exp.run(conf, port=8080 + 39)

    print("Top models:")
    for m in exp.export_top_models(formatter='dict'):
        print(m)

