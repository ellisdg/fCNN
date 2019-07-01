import os
import numpy as np
import nibabel as nib
import pandas as pd
from keras.models import load_model
from .utils.utils import load_json
from .utils.sequences import SubjectPredictionSequence
from .utils.hcp import new_cifti_scalar_like


def predict_subject(model, feature_filename, surface_filenames, surface_names, metric_names, output_filenames,
                    batch_size=50, window=(64, 64, 64), flip=False, spacing=(1, 1, 1), use_multiprocessing=False,
                    workers=1, max_queue_size=10, overwrite=False):
    for surface_filename, surface_name, output_filename in zip(surface_filenames, surface_names, output_filenames):
        if overwrite or not os.path.exists(output_filename):
            generator = SubjectPredictionSequence(feature_filename=feature_filename,
                                                  surface_filename=surface_filename,
                                                  surface_name=surface_name,
                                                  batch_size=batch_size,
                                                  window=window,
                                                  flip=flip,
                                                  spacing=spacing)
            prediction = model.predict_generator(generator,
                                                 use_multiprocessing=use_multiprocessing,
                                                 workers=workers,
                                                 max_queue_size=max_queue_size,
                                                 verbose=1)
            gifti_image = nib.gifti.GiftiImage()
            image_metadata = {'AnatomicalStructurePrimary': surface_name}
            gifti_image.meta.from_dict(image_metadata)
            for col, metric_name in enumerate(metric_names):
                metadata = {'Name': metric_name}
                darray = nib.gifti.GiftiDataArray(data=prediction[:, col], meta=metadata)
                gifti_image.add_gifti_data_array(darray)
            gifti_image.to_filename(output_filename)


def make_predictions(config_filename, model_filename, output_directory='./', n_subjects=None, shuffle=False,
                     key='validation_filenames', use_multiprocessing=False, n_workers=1, max_queue_size=5,
                     batch_size=50, output_replacements=('.func.gii', '.prediction.func.gii'), overwrite=False,
                     single_subject=None):
    output_directory = os.path.abspath(output_directory)
    config = load_json(config_filename)
    filenames = config[key]
    model_basename = os.path.basename(model_filename).replace(".h5", "")
    if single_subject is None:
        model = load_model(model_filename)
    else:
        model = None
    if n_subjects is not None:
        if shuffle:
            np.random.shuffle(filenames)
        filenames = filenames[:n_subjects]
    for feature_filename, surface_filenames, metric_filenames, subject_id in filenames:
        if single_subject is None or subject_id == single_subject:
            if model is None:
                model = load_model(model_filename)
            output_filenames = list()
            task = os.path.basename(metric_filenames[0]).split(".")[0]
            for hemisphere in config['hemispheres']:

                output_basename = "{task}.{hemi}.{model}_prediction.func.gii".format(hemi=hemisphere,
                                                                                      model=model_basename,
                                                                                      task=task)
                output_filenames.append(os.path.join(output_directory, output_basename))
            subject_metric_names = [metric_name.format(subject_id)
                                    for metric_name in np.squeeze(config['metric_names'])]
            predict_subject(model,
                            feature_filename,
                            surface_filenames,
                            config['surface_names'],
                            subject_metric_names,
                            output_filenames,
                            batch_size=batch_size,
                            window=np.asarray(config['window']),
                            spacing=np.asarray(config['spacing']),
                            flip=False,
                            overwrite=overwrite,
                            use_multiprocessing=use_multiprocessing,
                            workers=n_workers,
                            max_queue_size=max_queue_size)


def predict_local_subject(model, feature_filename, surface_filename, batch_size=50, window=(64, 64, 64),
                          spacing=(1, 1, 1), flip=False, use_multiprocessing=False, workers=1, max_queue_size=10,):
    generator = SubjectPredictionSequence(feature_filename=feature_filename, surface_filename=surface_filename,
                                          surface_name=None, batch_size=batch_size, window=window,
                                          flip=flip, spacing=spacing)
    return model.predict_generator(generator, use_multiprocessing=use_multiprocessing, workers=workers,
                                   max_queue_size=max_queue_size, verbose=1)


def whole_brain_scalar_predictions(model_filename, subject_ids, hcp_dir, output_dir, hemispheres, feature_basenames,
                                   surface_basename_template, target_basenames, model_name, n_outputs, n_features,
                                   window, criterion_name, metric_names, surface_names, reference, package="keras",
                                   n_gpus=1):
    from .scripts.run_trial import generate_hcp_filenames
    filenames = generate_hcp_filenames(directory=hcp_dir, surface_basename_template=surface_basename_template,
                                       target_basenames=target_basenames, feature_basenames=feature_basenames,
                                       subject_ids=subject_ids, hemispheres=hemispheres)
    if package == "pytorch":
        pytorch_whole_brain_scalar_predictions(model_filename=model_filename,
                                               model_name=model_name,
                                               n_outputs=n_outputs,
                                               n_features=n_features,
                                               filenames=filenames,
                                               prediction_dir=output_dir,
                                               window=window,
                                               criterion_name=criterion_name,
                                               metric_names=metric_names,
                                               surface_names=surface_names,
                                               reference=reference,
                                               n_gpus=n_gpus)
    else:
        raise ValueError("Predictions not yet implemented for {}".format(package))


def pytorch_whole_brain_scalar_predictions(model_filename, model_name, n_outputs, n_features, filenames, window,
                                           criterion_name, metric_names, surface_names, prediction_dir=None,
                                           output_csv=None, reference=None, n_gpus=1):
    from .train.pytorch import build_or_load_model
    from .utils.pytorch.dataset import WholeBrainCIFTI2DenseScalarDataset
    import torch.nn
    import torch

    model = build_or_load_model(model_name=model_name, model_filename=model_filename, n_outputs=n_outputs,
                                n_features=n_features, n_gpus=n_gpus)
    if reference is not None:
        reference = torch.from_numpy(reference).unsqueeze(0)
    basename = os.path.basename(model_filename).split(".")[0]
    if prediction_dir and not output_csv:
        output_csv = os.path.join(prediction_dir, str(basename) + "_prediction_scores.csv")
    dataset = WholeBrainCIFTI2DenseScalarDataset(filenames=filenames,
                                                 window=window,
                                                 metric_names=metric_names,
                                                 surface_names=surface_names,
                                                 spacing=None,
                                                 batch_size=1)
    criterion = getattr(torch.nn, criterion_name)()
    results = list()
    for args, idx in zip(dataset.filenames, range(len(dataset))):
        x, y = dataset[idx]
        subject_id = args[-1]
        prediction = model(x.unsqueeze(0))
        if n_gpus > 0:
            prediction = prediction.cpu()
        y = y.unsqueeze(0)
        error = criterion(prediction, y)
        row = [subject_id, error]
        if reference is not None:
            reference_error = criterion(reference, y)
            row.append(reference_error)
        results.append(row)
        if prediction_dir is not None:
            ref_filename = args[2][0][0]
            ref_basename = os.path.basename(ref_filename)
            ref_cifti = nib.load(ref_filename)
            _name = "_".join((subject_id, basename, "prediction"))
            _metric_names = [_metric_name.format(_name) for _metric_name in np.asarray(metric_names).ravel()]
            prediction_array = prediction.numpy().reshape(len(_metric_names),
                                                          np.sum(ref_cifti.get_axis(1).surface_mask))
            cifti_file = new_cifti_scalar_like(prediction_array, _metric_names, surface_names, ref_cifti)
            output_filename = os.path.join(prediction_dir, ref_basename.replace(subject_id, _name))
            cifti_file.to_filename(output_filename)

    if output_csv:
        columns = ["subject_id", criterion_name]
        if reference:
            columns.append("reference_" + criterion_name)
        pd.DataFrame(results, columns=columns).to_csv(output_csv)
