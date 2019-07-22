import os
import numpy as np
import nibabel as nib
import pandas as pd
from keras.models import load_model
from .utils.utils import load_json
from .utils.sequences import SubjectPredictionSequence
from .utils.hcp import new_cifti_scalar_like, get_metric_data


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
                                   n_gpus=1, n_workers=1, batch_size=1, model_kwargs=None):
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
                                               n_gpus=n_gpus,
                                               n_workers=n_workers,
                                               batch_size=batch_size,
                                               model_kwargs=model_kwargs)
    else:
        raise ValueError("Predictions not yet implemented for {}".format(package))


def pytorch_whole_brain_scalar_predictions(model_filename, model_name, n_outputs, n_features, filenames, window,
                                           criterion_name, metric_names, surface_names, prediction_dir=None,
                                           output_csv=None, reference=None, n_gpus=1, n_workers=1, batch_size=1,
                                           model_kwargs=None):
    from .train.pytorch import build_or_load_model, load_criterion
    from .utils.pytorch.dataset import WholeBrainCIFTI2DenseScalarDataset
    import torch
    from torch.utils.data import DataLoader

    if model_kwargs is None:
        model_kwargs = dict()

    model = build_or_load_model(model_name=model_name, model_filename=model_filename, n_outputs=n_outputs,
                                n_features=n_features, n_gpus=n_gpus, **model_kwargs)
    model.eval()
    basename = os.path.basename(model_filename).split(".")[0]
    if prediction_dir and not output_csv:
        output_csv = os.path.join(prediction_dir, str(basename) + "_prediction_scores.csv")
    dataset = WholeBrainCIFTI2DenseScalarDataset(filenames=filenames,
                                                 window=window,
                                                 metric_names=metric_names,
                                                 surface_names=surface_names,
                                                 spacing=None,
                                                 batch_size=1)
    criterion = load_criterion(criterion_name, n_gpus=n_gpus)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    results = list()
    with torch.no_grad():
        if reference is not None:
            reference = torch.from_numpy(reference).unsqueeze(0)
            if n_gpus > 0:
                reference = reference.cuda()
        for batch_idx, (x, y) in enumerate(loader):
            if n_gpus > 0:
                x = x.cuda()
                y = y.cuda()
            pred_y = model(x)
            if type(pred_y) == tuple:
                pred_y = pred_y[0]  # This is a hack to ignore other outputs that are used only for training
            for i in range(batch_idx):
                row = list()
                idx = (batch_idx * batch_size) + i
                args = dataset[idx]
                subject_id = args[-1]
                row.append(subject_id)
                idx_score = criterion(pred_y[i].unsqueeze(0), y[i].unsqueeze(0)).item()
                row.append(idx_score)
                if reference is not None:
                    idx_ref_score = criterion(reference.reshape(y[i].unsqueeze(0).shape),
                                              y[i].unsqueeze(0)).item()
                    row.append(idx_ref_score)
                results.append(row)
                save_predictions(prediction=pred_y[i].numpy(), args=args, basename=basename, metric_names=metric_names,
                                 surface_names=surface_names, prediction_dir=prediction_dir)

    if output_csv is not None:
        columns = ["subject_id", criterion_name]
        if reference is not None:
            columns.append("reference_" + criterion_name)
        pd.DataFrame(results, columns=columns).to_csv(output_csv)


def save_predictions(prediction, args, basename, metric_names, surface_names, prediction_dir):
    ref_filename = args[2][0]
    subject_id = args[-1]
    ref_basename = os.path.basename(ref_filename)
    prediction_name = "_".join((subject_id, basename, "prediction"))
    _metric_names = [_metric_name.format(prediction_name) for _metric_name in np.asarray(metric_names).ravel()]
    output_filename = os.path.join(prediction_dir, ref_basename.replace(subject_id, prediction_name))
    if prediction_dir is not None and not os.path.exists(output_filename):
        ref_cifti = nib.load(ref_filename)
        prediction_array = prediction.reshape(len(_metric_names),
                                                      np.sum(ref_cifti.header.get_axis(1).surface_mask))
        cifti_file = new_cifti_scalar_like(prediction_array, _metric_names, surface_names, ref_cifti)
        cifti_file.to_filename(output_filename)


def pytorch_subject_predictions(idx, model, dataset, criterion, basename, prediction_dir, surface_names, metric_names,
                                n_gpus, reference):
    import torch
    with torch.no_grad():
        args = dataset.filenames[idx]
        ref_filename = args[2][0]
        subject_id = args[-1]
        ref_basename = os.path.basename(ref_filename)
        prediction_name = "_".join((subject_id, basename, "prediction"))
        _metric_names = [_metric_name.format(prediction_name) for _metric_name in np.asarray(metric_names).ravel()]
        output_filename = os.path.join(prediction_dir, ref_basename.replace(subject_id, prediction_name))
        x, y = dataset[idx]
        if os.path.exists(output_filename):
            prediction = torch.from_numpy(get_metric_data([nib.load(output_filename)],
                                                          [_metric_names],
                                                          surface_names,
                                                          subject_id)).float().cpu()
        else:
            prediction = model(x.unsqueeze(0))
        if n_gpus > 0:
            prediction = prediction.cpu()
        y = y.unsqueeze(0)
        score = criterion(prediction.reshape(y.shape), y).item()
        row = [subject_id, score]
        if reference is not None:
            reference_score = criterion(reference.reshape(y.shape), y).item()
            row.append(reference_score)

        if prediction_dir is not None and not os.path.exists(output_filename):
            ref_cifti = nib.load(ref_filename)
            prediction_array = prediction.numpy().reshape(len(_metric_names),
                                                          np.sum(ref_cifti.header.get_axis(1).surface_mask))
            cifti_file = new_cifti_scalar_like(prediction_array, _metric_names, surface_names, ref_cifti)
            cifti_file.to_filename(output_filename)
    return row
