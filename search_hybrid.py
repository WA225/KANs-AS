import random
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import nni.retiarii.nn.pytorch as nnr
import nni.retiarii.strategy as strategy
import nni.retiarii.evaluator.pytorch.lightning as pl
from nni.retiarii import model_wrapper, serialize
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.serializer import basic_unit

# Import your real PyKAN KANLayer
from kan.KANLayer import KANLayer  
from kan.KANLayer import extend_grid

from pytorch_lightning.callbacks import Callback

from collections.abc import Iterable
import numpy as np

KANLayer = basic_unit(KANLayer)

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
# @serialize
class KANNet(nnr.Module):
    def __init__(self):
        super().__init__()
        # 5) number of hidden layers
        # self.num_hidden_layers = nnr.ValueChoice([1, 2, 3, 4], label='num_hidden_layers')
        # self.num_hidden_layers = 1
        # 2) neurons per layer (in/out dims for all hidden layers)
        # self.neurons_per_layer = nnr.ValueChoice([8, 32, 64], label='neurons_per_layer')
        # self.neurons_per_layer = [28*28, nnr.ValueChoice([8, 32, 64], label='neurons_per_layer_0'), nnr.ValueChoice([8, 32, 64], label='neurons_per_layer_1'), nnr.ValueChoice([8, 32, 64], label='neurons_per_layer_2'), nnr.ValueChoice([8, 32, 64], label='neurons_per_layer_3')]
        self.neurons_per_layer = [28*28, 32, 32, 10]
        # self.neurons_per_layer = 64
        # 1) number of spline intervals
        # self.num_spline_points = nnr.ValueChoice([5, 15, 25], label='num_spline_points')
        self.num_spline_points = 15
        # 2) spline polynomial order
        # self.spline_order = nnr.ValueChoice([1, 2, 3, 4], label='spline_order')
        self.spline_order = 3
        # 3) smoothness (scale_sp)
        # self.smoothness = nnr.ValueChoice([0.5, 1.0, 2.0], label='smoothness')
        self.smoothness = 1.0
        # 4) basis function
        # self.base_fun = nnr.LayerChoice([
        #     SiLU(),
        #     ReLU(),
        #     Tanh()
        # ], label='basis_function')
        
        # 5) grid range
        # self.grid_range = nnr.ValueChoice([
        #     (-1.0, 1.0),
        #     (-2.0, 2.0),
        #     (-3.0, 3.0),
        # ], label='grid_range')
        self.grid_range = (-3.0, 3.0)
        # 6) precision
        # self.precision = nnr.ValueChoice(['float32', 'float16'], label='precision')
        self.precision = 'float32'

        # Input layer
        # self.input_layer = KANLayer(
        #     in_dim=28*28,
        #     out_dim=self.neurons_per_layer[0],  # ValueChoice
        #     num=self.num_spline_points,
        #     k=self.spline_order,
        #     base_fun=self.base_fun,
        #     grid_range=self.grid_range,
        #     scale_sp=self.smoothness
        # )

        # Hidden stack (max 4), repeated per trial choice
        # 
        
        # def builder(layer_idx):
        #     return nnr.LayerChoice([
        #         KANLayer(
        #             in_dim=self.neurons_per_layer[layer_idx],
        #             out_dim=self.neurons_per_layer[layer_idx+1],
        #             num=nnr.ValueChoice([5, 15, 25], label=f'num_spline_{layer_idx}'),
        #             k=nnr.ValueChoice([1, 2, 3, 4], label=f'spline_order_{layer_idx}'),
        #             # base_fun=nnr.LayerChoice([SiLU(),ReLU(),Tanh()], label=f'basis_function_{layer_idx}'),
        #             grid_range=nnr.ValueChoice([(-1.0, 1.0),(-2.0, 2.0),(-3.0, 3.0),], label=f'grid_range_{layer_idx}'),
        #             # scale_sp=nnr.ValueChoice([0.5, 1.0, 2.0], label=f'smoothness_{layer_idx}')
        #             ),
        #         nnr.Linear(self.neurons_per_layer[layer_idx], self.neurons_per_layer[layer_idx+1])
        #     ], label=f'layer_choice_{layer_idx}')

        def builder(layer_idx):
            return nnr.LayerChoice([
                KANLayer(
                    in_dim=self.neurons_per_layer[layer_idx],
                    out_dim=self.neurons_per_layer[layer_idx+1],
                    num=self.num_spline_points,
                    k=self.spline_order,
                    # base_fun=nnr.LayerChoice([SiLU(),ReLU(),Tanh()], label=f'basis_function_{layer_idx}'),
                    grid_range=self.grid_range,
                    # scale_sp=nnr.ValueChoice([0.5, 1.0, 2.0], label=f'smoothness_{layer_idx}')
                    ),
                nnr.Linear(self.neurons_per_layer[layer_idx], self.neurons_per_layer[layer_idx+1])
            ], label=f'layer_choice_{layer_idx}')

        self.hidden_layers = nn.ModuleList([
            builder(i) for i in range(3)
        ])
        # self.selected_depth = nnr.ValueChoice([1, 2, 3, 4], label='hidden_depth')
        self.selected_depth = 3


        # self.out_layer = KANLayer(
        #     in_dim=self.neurons_per_layer[2],  # ValueChoice
        #     out_dim = 10,
        #     num=self.num_spline_points,
        #     k=self.spline_order,
        #     base_fun=self.base_fun,
        #     grid_range=self.grid_range,
        #     scale_sp=self.smoothness
        # )

    def forward(self, x):
        dtype = torch.float16 if self.precision == 'float16' else torch.float32
        x = x.view(x.size(0), -1).to(dtype=dtype)
        # x = self.input_layer(x)[0]

        # stack N hidden layers
        # for layer in self.hidden_layers:
        #     x = layer(x)
        last = False
        for i in range(self.selected_depth):
            x = self.hidden_layers[i](x)
            last = isinstance(x, tuple)
            if last:
                # print(x)
                x = x[0]

        # Dynamically set out_layer input dim based on final hidden layer output
        # final_dim = self.neurons_per_layer[self.selected_depth]
        # self.out_layer.in_dim = final_dim
        # return self.out_layer(x)[0]
        return x
    
    # def update_all_grids(self, samples_batch):
    #     # Update input layer (always KANLayer)
    #     new_grid = self.input_layer.update_grid_from_samples(samples_batch)
    #     if new_grid is not None:
    #         self.input_layer.grid = new_grid

    #     x = self.input_layer(samples_batch)[0]

    #     # Update hidden layers only if they are KANLayer
    #     for i in range(self.selected_depth):
    #         layer = self.hidden_layers[i]

    #         # If it's a LayerChoice, get the chosen layer
    #         chosen_layer = getattr(layer, 'choice', layer)

    #         # Only update if it's a KANLayer
    #         if isinstance(chosen_layer, KANLayer):
    #             new_grid = chosen_layer.update_grid_from_samples(x)
    #             if new_grid is not None:
    #                 chosen_layer.grid = new_grid

    #         x = layer(x)
    #         if isinstance(x, tuple):
    #             x = x[0]

    #     # Update output layer (KANLayer)
    #     new_grid = self.out_layer.update_grid_from_samples(x)
    #     if new_grid is not None:
    #         self.out_layer.grid = new_grid

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

    trainer = pl.Classification(
        train_dataloader=train_loader,
        val_dataloaders=val_loader,
        max_epochs=10
    )

    torch_rng_state, np_rng_state, py_rng_state = set_deterministic()
    model = KANNet()
    restore_random(torch_rng_state, np_rng_state, py_rng_state)

    strategy = strategy.TPEStrategy()
    exp = RetiariiExperiment(model, trainer, [], strategy)

    conf = RetiariiExeConfig('local')
    conf.experiment_name      = 'kan_mnist_search'
    conf.trial_concurrency    = 1
    conf.max_trial_number     = 10
    conf.training_service.use_active_gpu = False

    exp.run(conf, port=8080 + random.randint(0, 100))

    print("Top models:")
    for m in exp.export_top_models(formatter='dict'):
        print(m)

