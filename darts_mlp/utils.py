""" Utilities """
import os
import logging
import shutil
import torch
import numpy as np
import sklearn 
import torch.utils.data as data


def get_data(dataset_name, batch_size, num_workers):
    """ Get torchvision dataset """
    if dataset_name == 'california':
        dataset = sklearn.datasets.fetch_california_housing()
        X_cont = dataset["data"]
        Y = dataset["target"]
        X_cont = X_cont.astype(np.float32)
        n_cont_features = X_cont.shape[1]
        Y = Y.astype(np.float32)
        all_idx = np.arange(len(Y))

        trainval_idx, test_idx = sklearn.model_selection.train_test_split(
            all_idx, train_size=0.8, random_state=42
        )
        
        train_idx, val_idx = sklearn.model_selection.train_test_split(
            trainval_idx, train_size=0.8, random_state=42
        )

        X_train, y_train = X_cont[train_idx], Y[train_idx]
        X_val, y_val = X_cont[val_idx], Y[val_idx]
        X_test, y_test = X_cont[test_idx], Y[test_idx]

        noise = np.random.default_rng(42).normal(0.0, 1e-5, X_train.shape).astype(X_train.dtype)
        preprocessing = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
            output_distribution="normal",
            subsample=10**9,
        )
        preprocessing.fit(X_train + noise)

        X_train = preprocessing.transform(X_train)
        X_val = preprocessing.transform(X_val)
        X_test = preprocessing.transform(X_test)

        y_mean, y_std = y_train.mean(), y_train.std()
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

        dataset_train = data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        dataloader_train = data.DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

        dataset_val = data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        dataloader_val = data.DataLoader(
            dataset=dataset_val,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

        dataset_test = data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        dataloader_test = data.DataLoader(
            dataset=dataset_test,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

    elif dataset == 'covtype':
        dataset_name = sklearn.datasets.fetch_covtype()
        X_cont = dataset["data"][:, :10]
        X_cat = dataset["data"][:, 10:]
        Y = dataset["target"]
        X_cont = X_cont.astype(np.float32)
        X_cat = X_cat.astype(np.int64)
        n_cont_features = X_cont.shape[1]
        Y = Y.astype(np.int64) - 1

        all_idx = np.arange(len(Y))
        trainval_idx, test_idx = sklearn.model_selection.train_test_split(
            all_idx, train_size=0.8, random_state=42
        )
        train_idx, val_idx = sklearn.model_selection.train_test_split(
            trainval_idx, train_size=0.8, random_state=42
        )

        X_cont_train, X_cat_train, y_train = X_cont[train_idx], X_cat[train_idx], Y[train_idx]
        X_cont_val, X_cat_val, y_val = X_cont[val_idx], X_cat[val_idx], Y[val_idx]
        X_cont_test, X_cat_test, y_test = X_cont[test_idx], X_cat[test_idx], Y[test_idx]

        noise = np.random.default_rng(42).normal(0.0, 1e-5, X_cont_train.shape).astype(X_cont_train.dtype)
        preprocessing = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
            output_distribution="normal",
            subsample=10**9,
        )
        preprocessing.fit(X_cont_train + noise)

        X_cont_train = preprocessing.transform(X_cont_train)
        X_cont_val = preprocessing.transform(X_cont_val)
        X_cont_test = preprocessing.transform(X_cont_test)

        dataset_train = data.TensorDataset(torch.tensor(X_cont_train), torch.tensor(X_cat_train), torch.tensor(y_train))
        dataloader_train = data.DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

        dataset_val = data.TensorDataset(torch.tensor(X_cont_val), torch.tensor(X_cat_val), torch.tensor(y_val))
        dataloader_val = data.DataLoader(
            dataset=dataset_val,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

        dataset_test = data.TensorDataset(torch.tensor(X_cont_test), torch.tensor(X_cat_test), torch.tensor(y_test))
        dataloader_test = data.DataLoader(
            dataset=dataset_test,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

    return {
    'train': (dataloader_train, n_cont_features),
    'val': dataloader_val,
    'test': dataloader_test
}


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)
