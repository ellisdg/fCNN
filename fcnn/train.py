import os
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from .resnet import load_model, ResnetBuilder
import numpy as np
from .utils.utils import load_json
from .utils.sequences import HCPRegressionSequence


def run_training(config_filename, model_filename, training_log_filename, verbose=1, use_multiprocessing=False,
                 n_workers=1, max_queue_size=5, model_name='resnet_34'):
    """
    :param model_name:
    :param verbose:
    :param use_multiprocessing:
    :param n_workers:
    :param max_queue_size:
    :param config_filename:
    :param model_filename:
    :param training_log_filename:
    :return:

    Anything that directly affects the training results should go into the config file. Other specifications such as
    multiprocessing optimization should be arguments to this function, as these arguments affect the computation time,
    but the results should not vary based on whether multiprocessing is used or not.
    """
    config = load_json(config_filename)
    window = np.asarray(config['window'])
    spacing = np.asarray(config['spacing'])

    if 'model_name' in config:
        model_name = config['model_name']

    # 2. Create model
    if os.path.exists(model_filename):
        model = load_model(model_filename)
    else:
        input_shape = tuple(window.tolist() + [config['n_features']])
        model = getattr(ResnetBuilder, 'build_' + model_name)(input_shape, len(config['metric_names']),
                                                              activation=config['activation'])
        model.compile(optimizer=config['optimizer'], loss=config['loss'])

    # 4. Create Generators

    training_generator = HCPRegressionSequence(filenames=config['training_filenames'],
                                               batch_size=config['batch_size'],
                                               flip=config['flip'],
                                               reorder=config['reorder'],
                                               window=window,
                                               spacing=spacing,
                                               points_per_subject=config['points_per_subject'],
                                               surface_names=config['surface_names'],
                                               metric_names=config['metric_names'])

    validation_generator = HCPRegressionSequence(filenames=config['validation_filenames'],
                                                 batch_size=config['validation_batch_size'],
                                                 flip=False,
                                                 reorder=config['reorder'],
                                                 window=window,
                                                 spacing=spacing,
                                                 points_per_subject=config['validation_points_per_subject'],
                                                 surface_names=config['surface_names'],
                                                 metric_names=config['metric_names'])

    # 5. Run Training

    checkpointer = ModelCheckpoint(filepath=model_filename,
                                   verbose=verbose,
                                   save_best_only=config['save_best_only'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=config['decay_factor'],
                                  patience=config['decay_patience'],
                                  min_lr=config['min_learning_rate'])
    csv_logger = CSVLogger(training_log_filename, append=True)
    history = model.fit_generator(generator=training_generator,
                                  epochs=config['n_epochs'],
                                  use_multiprocessing=use_multiprocessing,
                                  workers=n_workers,
                                  max_queue_size=max_queue_size,
                                  callbacks=[checkpointer, reduce_lr, csv_logger],
                                  validation_data=validation_generator)
