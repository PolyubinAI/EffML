import utils
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm.auto import tqdm
import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import functools

def train_step(model, optimizer, dataloader, loss_fn, device, task_type):
    model.train()
    losses = []
    for batch in dataloader:
        if len(batch) == 3:
            x_cont, x_cat, y = batch
            x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)
        else:
            x_cont, y = batch
            x_cont, y = x_cont.to(device), y.to(device)
            x_cat = None
        optimizer.zero_grad()
        preds = model(x_cont, x_cat)
        if task_type == 'regression':
            preds = preds.flatten()
        loss = loss_fn(preds, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(model, dataloader, device, metric_func, task_type):
    model.eval()
    preds_all = []
    y_all = []
    for batch in dataloader:
        if len(batch) == 3:
            x_cont, x_cat, y = batch
            x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)
        if len(batch) == 2:
            x_cont, y = batch
            x_cont, y = x_cont.to(device), y.to(device)
            x_cat = None
        preds = model(x_cont, x_cat)
        if task_type == 'regression':
            preds = preds.flatten()
        preds_all.append(preds.cpu().detach())
        y_all.append(y.cpu().detach())
    preds_all = torch.cat(preds_all)
    if task_type == 'classification':
        preds_all = preds_all.argmax(dim=1)
    preds_all = preds_all.numpy()
    y_all = torch.cat(y_all).numpy()
    return metric_func(y_all, preds_all)


def run_experiment(model, optimizer, dataloaders, loss_fn, device, metric_func, 
                   task_type, n_epoches, maximize=True, y_std=None):
    model.to(device)
    best_val = None
    best_test = None
    best_step = None
    for i in (pbar := tqdm(range(1, n_epoches + 1))):
        loss = train_step(model, optimizer, dataloaders['train'], loss_fn, device, task_type)
        metric_val = evaluate(model, dataloaders['val'], device, metric_func, task_type)
        metric_test = evaluate(model, dataloaders['test'], device, metric_func, task_type)
        str_desc = f'Loss: {loss}, val metric {metric_val}, test metric {metric_test}'
        pbar.set_description(str_desc)
        if best_val is None or (maximize and metric_val > best_val) or (not maximize and metric_val < best_val):
            best_val = metric_val
            best_test = metric_test
            best_step = i
    if task_type == 'regression' and y_std:
        best_val *= y_std
        best_test *= y_std
    return {
        'model': model.cpu(),
        'val': best_val,
        'test': best_test,
        'best_step': best_step
    }

def experiments_series(experiment_name, model_class, model_args, model_kwargs,
                       optimizer_class, learning_rate,
                       dataloaders, loss_fn, device, metric_func, 
                       task_type, n_epoches, maximize, n_runs):
    print(f'===== Running experiment "{experiment_name}" =====')
    val_results = []
    test_results = []
    for _ in tqdm(range(n_runs)):
        model = model_class(*model_args, **model_kwargs)
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)
        results = run_experiment(model, optimizer, dataloaders, loss_fn, device, metric_func,
                                task_type, n_epoches, maximize)
        val_results.append(results['val'])
        test_results.append(results['test'])
    val_results = np.array(val_results)
    test_results = np.array(test_results)
    
    print('===== Experiments results =====')
    print(f'Validation metric: {val_results.mean()}±{val_results.std()}')
    print(f'Corresponding test metric: {test_results.mean()}±{test_results.std()}')

class MLP(nn.Module):
    def __init__(self, in_size, out_size, activation, dropout=False, affine=True):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(in_size, out_size),
        activation,
        nn.BatchNorm1d(out_size, affine=affine)
        )
    def forward(self, x):
        return self.net(x)

class BestNASModel1(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.first_layer = MLP(in_size=in_size, out_size=out_size, activation=nn.Tanh())
        self.second_layer = MLP(in_size=out_size, out_size=out_size, activation=nn.ReLU())
        self.last_layer = nn.Linear(out_size, 1)

    def forward(self, x, x_cat=[]):
        x = self.first_layer(x)
        x = self.second_layer(x)
        x = self.last_layer(x)
        return x
    
class BestNASModel2(nn.Module):
    def __init__(self, in_size, out_size, cat_cardinalities):
        super().__init__()
        self.cat_cardinalities = cat_cardinalities
        self.d_cat = sum(cat_cardinalities)

        self.first_layer = MLP(in_size=in_size + self.d_cat, out_size=out_size, activation=nn.Tanh())
        self.last_layer = nn.Linear(out_size, 7)

    def forward(self, x, x_cat=[]):
        x = x + x_cat
        x = self.first_layer(x)
        x = self.last_layer(x)
        return x
    

def check():
    dataloaders = utils.get_data(dataset_name='california', batch_size=256, num_workers=0)
    dataloaders = {
        'train': dataloaders['train'][0],
        'val': dataloaders['val'],
        'test': dataloaders['test'],
    }

    device = torch.device('cuda:1')

    experiments_series(
    experiment_name='Nas model California Housing data',
    model_class=BestNASModel1,
    model_args=[],
    model_kwargs={
        'in_size':  8,
        'out_size': 384,
    },
    optimizer_class=torch.optim.Adam,
    learning_rate=3e-4,
    dataloaders=dataloaders,
    loss_fn=F.mse_loss,
    device=device,
    metric_func=functools.partial(sklearn.metrics.mean_squared_error, squared=False),
    task_type='regression',
    n_epoches=100,
    maximize=False,
    n_runs=10
)


if __name__ == "__main__":
    check()
    
