import os
import numpy as np
import nibabel as nib
from keras.models import load_model
from .utils.utils import load_json
from .utils.sequences import SubjectPredictionSequence


def predict_subject(model, feature_filename, surface_filenames, surface_names, metric_names, output_filenames,
                    batch_size=50, window=(64, 64, 64), flip=False, spacing=(1, 1, 1), use_multiprocessing=False,
                    workers=1, max_queue_size=10):
    for surface_filename, surface_name, output_filename in zip(surface_filenames, surface_names, output_filenames):
        generator = SubjectPredictionSequence(feature_filename=feature_filename, surface_filename=surface_filename,
                                              surface_name=surface_name, batch_size=batch_size, window=window,
                                              flip=flip, spacing=spacing)
        prediction = model.predict_generator(generator, use_multiprocessing=use_multiprocessing, workers=workers,
                                             max_queue_size=max_queue_size, verbose=1)
        gifti_image = nib.gifti.GiftiImage()
        for col, metric_name in enumerate(metric_names):
            metadata = {'AnatomicalStructurePrimary': surface_name,
                        'Name': metric_name}
            darray = nib.gifti.GiftiDataArray(data=prediction[:, col], meta=metadata)
            gifti_image.add_gifti_data_array(darray)
        gifti_image.to_filename(output_filename)


def make_predictions(config_filename, model_filename, output_directory='./', n_subjects=None, shuffle=False,
                     key='validation_filenames', use_multiprocessing=False, n_workers=1, max_queue_size=5,
                     batch_size=50, output_replacements=('.func.gii', '.prediction.func.gii')):
    output_directory = os.path.abspath(output_directory)
    config = load_json(config_filename)
    filenames = config[key]
    model = load_model(model_filename)
    if n_subjects is not None:
        if shuffle:
            np.random.shuffle(filenames)
        filenames = filenames[:n_subjects]
    for feature_filename, surface_filenames, metric_filenames, subject_id in filenames:
        output_filenames = [os.path.join(output_directory,
                                         os.path.basename(filename).replace(*output_replacements))
                            for filename in metric_filenames]
        subject_metric_names = [metric_name.format(subject_id) for metric_name in config['metric_names']]
        predict_subject(model, feature_filename, surface_filenames, config['surface_names'], subject_metric_names,
                        output_filenames, batch_size=batch_size, window=np.asarray(config['window']),
                        spacing=np.asarray(config['spacing']), flip=False,
                        use_multiprocessing=use_multiprocessing, workers=n_workers, max_queue_size=max_queue_size)
