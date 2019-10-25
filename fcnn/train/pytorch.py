import os
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn

from ..utils.pytorch import WholeBrainCIFTI2DenseScalarDataset
from ..models.pytorch import fetch_model_by_name
from .pytorch_training_utils import epoch_training, epoch_validatation, collate_flatten
from ..utils.pytorch import functions


def build_or_load_model(model_name, model_filename, n_features, n_outputs, n_gpus=0, bias=None, freeze_bias=False,
                        **kwargs):
    model = fetch_model_by_name(model_name, n_features=n_features, n_outputs=n_outputs, **kwargs)
    if bias is not None:
        model.fc.bias = torch.nn.Parameter(torch.from_numpy(bias))
    if freeze_bias:
        model.fc.bias.requires_grad_(False)
    if n_gpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    elif n_gpus > 0:
        model = model.cuda()
    if os.path.exists(model_filename):
        model.load_state_dict(torch.load(model_filename))
    return model


def build_optimizer(optimizer_name, model_parameters, learning_rate=1e-4):
    return getattr(torch.optim, optimizer_name)(model_parameters, lr=learning_rate)


def run_pytorch_training(config, model_filename, training_log_filename, verbose=1, use_multiprocessing=False,
                         n_workers=1, max_queue_size=5, model_name='resnet_34', n_gpus=1, regularized=False,
                         sequence_class=WholeBrainCIFTI2DenseScalarDataset, directory=None, test_input=1,
                         metric_to_monitor="loss", model_metrics=(), bias=None,  **unused_args):
    """
    :param test_input: integer with the number of inputs from the generator to write to file. 0, False, or None will
    write no inputs to file.
    :param sequence_class: class to use for the generator sequence
    :param model_name:
    :param verbose:
    :param use_multiprocessing:
    :param n_workers:
    :param max_queue_size:
    :param config:
    :param model_filename:
    :param training_log_filename:
    :param metric_to_monitor:
    :param model_metrics:
    :return:

    Anything that directly affects the training results should go into the config file. Other specifications such as
    multiprocessing optimization should be arguments to this function, as these arguments affect the computation time,
    but the results should not vary based on whether multiprocessing is used or not.
    """
    window = np.asarray(config['window'])
    spacing = np.asarray(config['spacing'])
    if 'model_name' in config:
        model_name = config['model_name']

    if "n_outputs" in config:
        n_outputs = config['n_outputs']
    else:
        n_outputs = len(np.concatenate(config['metric_names']))

    if "model_kwargs" in config:
        model_kwargs = config["model_kwargs"]
    else:
        model_kwargs = dict()

    if "freeze_bias" in config:
        freeze_bias = config["freeze_bias"]
    else:
        freeze_bias = False

    model = build_or_load_model(model_name, model_filename, n_features=config["n_features"], n_outputs=n_outputs,
                                freeze_bias=freeze_bias, bias=bias, n_gpus=n_gpus, **model_kwargs)
    model.train()

    criterion = load_criterion(config['loss'], n_gpus=n_gpus)

    if "regularized" in config:
        regularized = config["regularized"]

    optimizer_kwargs = dict()
    if "initial_learning_rate" in config:
        optimizer_kwargs["learning_rate"] = config["initial_learning_rate"]

    optimizer = build_optimizer(optimizer_name=config["optimizer"],
                                model_parameters=model.parameters(),
                                **optimizer_kwargs)

    if "iterations_per_epoch" in config:
        iterations_per_epoch = config["iterations_per_epoch"]
    else:
        iterations_per_epoch = 1
    if "additional_training_args" in config:
        train_kwargs = config["additional_training_args"]
    else:
        train_kwargs = dict()

    if "sequence_kwargs" in config:
        sequence_kwargs = config["sequence_kwargs"]
    else:
        sequence_kwargs = dict()

    print("train_kwargs", train_kwargs)
    print("sequence_kwargs", sequence_kwargs)

    # 4. Create datasets
    training_dataset = sequence_class(filenames=config['training_filenames'],
                                      flip=config['flip'],
                                      reorder=config['reorder'],
                                      window=window,
                                      spacing=spacing,
                                      points_per_subject=config['points_per_subject'],
                                      surface_names=config['surface_names'],
                                      metric_names=config['metric_names'],
                                      base_directory=directory,
                                      subject_ids=config["training"],
                                      **train_kwargs,
                                      **sequence_kwargs)

    training_loader = DataLoader(training_dataset,
                                 batch_size=config["batch_size"],
                                 shuffle=True,
                                 num_workers=n_workers,
                                 collate_fn=collate_flatten)

    if test_input:
        for index in range(test_input):
            x, y = training_dataset[index]
            x_image = nib.Nifti1Image(x.numpy()[index], affine=np.diag(np.ones(4)))
            x_image.to_filename(model_filename.replace(".pt",
                                                       "_input_test_{}.nii.gz".format(index)))

    if 'skip_validation' in config and config['skip_validation']:
        validation_loader = None
        metric_to_monitor = "loss"
    else:
        validation_dataset = sequence_class(filenames=config['validation_filenames'],
                                            flip=False,
                                            reorder=config['reorder'],
                                            window=window,
                                            spacing=spacing,
                                            points_per_subject=config['validation_points_per_subject'],
                                            surface_names=config['surface_names'],
                                            metric_names=config['metric_names'],
                                            **sequence_kwargs)
        validation_loader = DataLoader(validation_dataset,
                                       batch_size=config["validation_batch_size"],
                                       shuffle=False,
                                       num_workers=n_workers,
                                       collate_fn=collate_flatten)

    train(model=model, optimizer=optimizer, criterion=criterion, n_epochs=config["n_epochs"], verbose=bool(verbose),
          training_loader=training_loader, validation_loader=validation_loader, model_filename=model_filename,
          training_log_filename=training_log_filename, iterations_per_epoch=iterations_per_epoch,
          metric_to_monitor=metric_to_monitor, early_stopping_patience=config["early_stopping_patience"],
          save_best_only=config["save_best_only"], learning_rate_decay_patience=config["decay_patience"],
          regularized=regularized, n_gpus=n_gpus)


def train(model, optimizer, criterion, n_epochs, training_loader, validation_loader, training_log_filename,
          model_filename, iterations_per_epoch=1, metric_to_monitor="val_loss", early_stopping_patience=None,
          learning_rate_decay_patience=None, save_best_only=False, n_gpus=1, verbose=True, regularized=False):
    training_log = list()
    if os.path.exists(training_log_filename):
        training_log.extend(pd.read_csv(training_log_filename).values)
    training_log_header = ["epoch", "loss", "lr", "val_loss"]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=verbose)

    for epoch in range(n_epochs):

        # early stopping
        if (training_log and early_stopping_patience
            and np.asarray(training_log)[:, training_log_header.index(metric_to_monitor)].argmin()
                <= len(training_log) - early_stopping_patience):
            print("Early stopping patience {} has been reached.".format(early_stopping_patience))
            break

        # train the model
        loss = epoch_training(training_loader, model, criterion, optimizer=optimizer, epoch=epoch, gpu=n_gpus,
                              regularized=regularized)

        # predict validation data
        if validation_loader:
            val_loss = epoch_validatation(validation_loader, model, criterion, gpu=n_gpus, regularized=regularized)
        else:
            val_loss = None

        # update the training log
        training_log.append([epoch, loss, get_lr(optimizer), val_loss])
        pd.DataFrame(training_log, columns=training_log_header).set_index("epoch").to_csv(training_log_filename)
        min_epoch = np.asarray(training_log)[:, training_log_header.index(metric_to_monitor)].argmin()

        # check loss and decay
        if learning_rate_decay_patience:
            if validation_loader:
                scheduler.step(val_loss)
            else:
                scheduler.step(loss)

        # save model
        if not save_best_only or min_epoch == len(training_log) - 1:
            torch.save(model.state_dict(), model_filename)


def get_lr(optimizer):
    lrs = [params['lr'] for params in optimizer.param_groups]
    return np.squeeze(np.unique(lrs))


def load_criterion(criterion_name, n_gpus=0):
    try:
        criterion = getattr(functions, criterion_name)
    except AttributeError:
        criterion = getattr(torch.nn, criterion_name)()
        if n_gpus > 0:
            criterion.cuda()
    return criterion
