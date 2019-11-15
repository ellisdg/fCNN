import os
import glob
import numpy as np
import nibabel as nib
from keras.utils import Sequence
from unet3d.utils.nilearn_custom_utils.nilearn_utils import crop_img, reorder_affine
from unet3d.utils.utils import resize_affine, resample
from unet3d.augment import scale_affine, add_noise
from unet3d.data import combine_images

from .radiomic_utils import binary_classification, multilabel_classification, fetch_data, pick_random_list_elements, \
    fetch_data_for_point
from .radiomic_utils import load_single_image
from .hcp import nib_load_files, extract_gifti_surface_vertices, get_vertices_from_scalar, get_metric_data
from .utils import (read_polydata, extract_polydata_vertices, zero_mean_normalize_image_data,
                    zero_floor_normalize_image_data, zero_one_window, compile_one_hot_encoding)


def load_image(filename, feature_axis=3, resample_unequal_affines=True, interpolation="linear", force_4d=False):
    """
    :param feature_axis: axis along which to combine the images, if necessary.
    :param filename: can be either string path to the file or a list of paths.
    :return: image containing either the 1 image in the filename or a combined image based on multiple filenames.
    """

    if type(filename) != list:
        if not force_4d:
            return nib.load(filename)
        else:
            filename = [filename]

    return combine_images(nib_load_files(filename), axis=feature_axis,
                          resample_unequal_affines=resample_unequal_affines, interpolation=interpolation)


class SingleSiteSequence(Sequence):
    def __init__(self, filenames, batch_size, target_labels, window, spacing, classification='binary', shuffle=True,
                 points_per_subject=1, flip=False, reorder=False, iterations_per_epoch=1,
                 deformation_augmentation=None, base_directory=None, subject_ids=None):
        self.deformation_augmentation = deformation_augmentation
        self.base_directory = base_directory
        self.subject_ids = subject_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filenames = filenames
        self.target_labels = target_labels
        self.window = window
        self.points_per_subject = points_per_subject
        self.flip = flip
        self.reorder = reorder
        self.spacing = spacing
        self.iterations_per_epoch = iterations_per_epoch
        self.subjects_per_batch = int(np.floor(self.batch_size / self.points_per_subject))
        assert self.subjects_per_batch > 0
        if classification == 'binary':
            self._classify = binary_classification
        elif classification == 'multilabel':
            self._classify = multilabel_classification
        else:
            self._classify = classification
        self.generate_epoch_filenames()

    def get_number_of_subjects_per_epoch(self):
        return self.get_number_of_subjects() * self.iterations_per_epoch

    def get_number_of_subjects(self):
        return len(self.filenames)

    def generate_epoch_filenames(self):
        _filenames = list(self.filenames)
        epoch_filenames = list()
        for i in range(self.iterations_per_epoch):
            if self.shuffle:
                np.random.shuffle(_filenames)
            epoch_filenames.extend(_filenames)
        self.epoch_filenames = list(epoch_filenames)

    def switch_to_augmented_filename(self, subject_id, filename):
        augmented_filename = self.deformation_augmentation.format(base_directory=self.base_directory,
                                                                  random_subject_id=np.random.choice(self.subject_ids),
                                                                  subject_id=subject_id,
                                                                  basename=os.path.basename(filename))
        if not os.path.exists(augmented_filename):
            raise RuntimeWarning("Augmented filename {} does not exists!".format(augmented_filename))
        else:
            filename = augmented_filename
        return filename

    def __len__(self):
        return self.get_number_of_batches_per_epoch()

    def get_number_of_batches_per_epoch(self):
        return int(np.floor(np.divide(self.get_number_of_subjects_per_epoch(),
                                      self.subjects_per_batch)))

    def __getitem__(self, idx):
        batch_filenames = self.epoch_filenames[idx * self.subjects_per_batch:(idx + 1) * self.subjects_per_batch]
        batch_x = list()
        batch_y = list()
        for feature_filename, target_filename in batch_filenames:
            for x, y in zip(*fetch_data(feature_filename,
                                        target_filename,
                                        self.target_labels,
                                        self.window,
                                        n_points=self.points_per_subject,
                                        flip=self.flip,
                                        reorder=self.reorder,
                                        spacing=self.spacing,
                                        classify=self._classify)):
                batch_x.append(x)
                batch_y.append(y)
        return np.asarray(batch_x), np.asarray(batch_y)

    def on_epoch_end(self):
        self.generate_epoch_filenames()


class MultiScannerSequence(SingleSiteSequence):
    def __init__(self, filenames, supplemental_filenames, batch_size, n_supplemental, target_labels, window, spacing,
                 classification='binary', points_per_subject=1, flip=False, reorder=False):
        self.n_supplemental = n_supplemental
        self.supplemental_filenames = supplemental_filenames
        super().__init__(filenames, batch_size, target_labels, window, spacing, classification, points_per_subject,
                         flip, reorder)

    def get_number_of_subjects_per_epoch(self):
        return len(self.filenames) + self.n_supplemental

    def generate_epoch_filenames(self):
        supplemental = pick_random_list_elements(self.supplemental_filenames,
                                                 self.n_supplemental,
                                                 replace=False)
        epoch_filenames = list(self.filenames) + list(supplemental)
        np.random.shuffle(epoch_filenames)
        self.epoch_filenames = list(epoch_filenames)


class HCPRegressionSequence(SingleSiteSequence):
    def __init__(self, filenames, batch_size, window, spacing, metric_names, classification=None,
                 surface_names=('CortexLeft', 'CortexRight'), **kwargs):
        super().__init__(filenames=filenames, batch_size=batch_size, target_labels=tuple(), window=window,
                         spacing=spacing, classification=classification, **kwargs)
        self.metric_names = metric_names
        self.surface_names = surface_names

    def __getitem__(self, idx):
        return self.fetch_hcp_regression_batch(idx)

    def fetch_hcp_regression_batch(self, idx):
        batch_filenames = self.epoch_filenames[idx * self.subjects_per_batch:(idx + 1) * self.subjects_per_batch]
        batch_x = list()
        batch_y = list()
        for args in batch_filenames:
            _x, _y = self.fetch_hcp_subject_batch(*args)
            batch_x.extend(_x)
            batch_y.extend(_y)
        return np.asarray(batch_x), np.asarray(batch_y)

    def fetch_hcp_subject_batch(self, feature_filename, surface_filenames, metric_filenames, subject_id):
        metrics, all_metric_data = self.load_metric_data(metric_filenames, subject_id)
        vertices = self.extract_vertices(surface_filenames, metrics)
        random_vertices, random_target_values = self.select_random_vertices_and_targets(vertices, all_metric_data)
        return self.load_feature_data(feature_filename, random_vertices, random_target_values)

    def load_metric_data(self, metric_filenames, subject_id):
        metrics = nib_load_files(metric_filenames)
        all_metric_data = get_metric_data(metrics, self.metric_names, self.surface_names, subject_id)
        return metrics, all_metric_data

    def load_feature_data(self, feature_filename, random_vertices, random_target_values):
        batch_x = list()
        batch_y = list()
        feature_image = load_single_image(feature_filename, reorder=self.reorder)
        for vertex, y in zip(random_vertices, random_target_values):
            x = fetch_data_for_point(vertex, feature_image, window=self.window, flip=self.flip, spacing=self.spacing)
            batch_x.append(x)
            batch_y.append(y)
        return batch_x, batch_y

    def select_random_vertices_and_targets(self, vertices, all_metric_data):
        # randomly select the target vertices and corresponding values
        indices = np.random.choice(np.arange(vertices.shape[0]), size=self.points_per_subject, replace=False)
        random_vertices = vertices[indices]
        random_target_values = all_metric_data[indices]
        return random_vertices, random_target_values

    def extract_vertices(self, surface_filenames, metrics):
        # extract the vertices
        surfaces = nib_load_files(surface_filenames)
        vertices = list()
        for surface, surface_name in zip(surfaces, self.surface_names):
            vertices_index = get_vertices_from_scalar(metrics[0], brain_structure_name=surface_name)
            surface_vertices = extract_gifti_surface_vertices(surface, primary_anatomical_structure=surface_name)
            vertices.extend(surface_vertices[vertices_index])
        return np.asarray(vertices)


class ParcelBasedSequence(HCPRegressionSequence):
    def __init__(self, *args, target_parcel, parcellation_template, parcellation_name, **kwargs):
        self.target_parcel = target_parcel
        self.parcellation_template = parcellation_template
        self.parcellation_name = parcellation_name
        super().__init__(*args, **kwargs)

    def fetch_hcp_subject_batch(self, feature_filename, surface_filenames, metric_filenames, subject_id):
        metrics, all_metric_data = self.load_metric_data(metric_filenames, subject_id)
        vertices = self.extract_vertices(surface_filenames, metrics)
        parcellation = self.load_parcellation(subject_id)
        parcellation_mask = np.in1d(parcellation, self.target_parcel)
        random_vertices, random_target_values = self.select_random_vertices_and_targets(
            vertices[parcellation_mask], all_metric_data[parcellation_mask])
        return self.load_feature_data(feature_filename, random_vertices, random_target_values)

    def load_parcellation(self, subject_id):
        parcellation_filename = self.parcellation_template.format(subject_id)
        parcellation = np.squeeze(get_metric_data(metrics=nib_load_files([parcellation_filename]),
                                                  metric_names=[[self.parcellation_name]],
                                                  surface_names=self.surface_names,
                                                  subject_id=subject_id))
        return parcellation


class SubjectPredictionSequence(Sequence):
    def __init__(self, feature_filename, surface_filename, surface_name, batch_size=50, window=(64, 64, 64), flip=False,
                 spacing=(1, 1, 1)):
        self.feature_image = load_image(feature_filename)
        if ".gii" in surface_filename:
            surface = nib.load(surface_filename)
            self.vertices = extract_gifti_surface_vertices(surface, primary_anatomical_structure=surface_name)
        elif ".vtk" in surface_filename:
            surface = read_polydata(surface_filename)
            self.vertices = extract_polydata_vertices(surface)
        else:
            raise RuntimeError("Uknown file type: ", surface_filename)
        self.batch_size = batch_size
        self.window = window
        self.flip = flip
        self.spacing = spacing

    def __len__(self):
        return int(np.ceil(np.divide(len(self.vertices), self.batch_size)))

    def __getitem__(self, idx):
        batch_vertices = self.vertices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [fetch_data_for_point(vertex,
                                      self.feature_image,
                                      window=self.window,
                                      flip=self.flip,
                                      spacing=self.spacing) for vertex in batch_vertices]
        return np.asarray(batch)


class WholeBrainRegressionSequence(HCPRegressionSequence):
    def __init__(self, resample='linear', crop=True, augment_scale_std=0, additive_noise_std=0,
                 normalization="zero_mean", **kwargs):
        super().__init__(**kwargs)
        self.resample = resample
        self.crop = crop
        self.augment_scale_std = augment_scale_std
        self.additive_noise_std = additive_noise_std
        if normalization == "zero_mean":
            self.normalization_func = zero_mean_normalize_image_data
        elif normalization == "zero_floor":
            self.normalization_func = zero_floor_normalize_image_data
        elif normalization == "zero_one_window":
            self.normalization_func = zero_one_window
        else:
            self.normalization_func = lambda x, **kwargs: x

    def __len__(self):
        return int(np.ceil(np.divide(len(self.filenames) * self.iterations_per_epoch, self.batch_size)))

    def __getitem__(self, idx):
        x = list()
        y = list()
        batch_filenames = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        for feature_filename, surface_filenames, metric_filenames, subject_id in batch_filenames:
            if self.deformation_augmentation:
                feature_filename = self.switch_to_augmented_filename(subject_id=subject_id, filename=feature_filename)
            metrics = nib_load_files(metric_filenames)
            x.append(self.resample_input(feature_filename))
            y.append(get_metric_data(metrics, self.metric_names, self.surface_names, subject_id).T.ravel())
        return np.asarray(x), np.asarray(y)

    def resample_input(self, feature_filename):
        feature_image = load_image(feature_filename)
        affine = feature_image.affine.copy()
        shape = feature_image.shape
        if self.reorder:
            affine = reorder_affine(affine, shape)
        if self.crop:
            affine, shape = crop_img(feature_image, return_affine=True, pad=True)
        if self.augment_scale_std:
            scale = np.random.normal(1, self.augment_scale_std, 3)
            affine = scale_affine(affine, shape, scale)
        if self.additive_noise_std:
            feature_image.get_data()[:] = add_noise(feature_image.get_data(), sigma_factor=self.additive_noise_std)
        affine = resize_affine(affine, shape, self.window)
        input_img = resample(feature_image, affine, self.window, interpolation=self.resample)
        return self.normalization_func(input_img.get_data())


class WholeBrainAutoEncoder(WholeBrainRegressionSequence):
    def __getitem__(self, idx):
        x_batch = list()
        y_batch = list()
        batch_filenames = self.epoch_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        for item in batch_filenames:
            x, y = self.resample_input(item)
            x_batch.append(x)
            y_batch.append(y)
        return np.asarray(x_batch), np.asarray(y_batch)

    def resample_input(self, input, normalize=True):
        input_image, target_image = self.resample_image(input, normalize=normalize)
        return input_image.get_data(), target_image.get_data()

    def resample_image(self, input, normalize=True, feature_index=0, target_index=None, target_resample=None):
        feature_filename = input[feature_index]
        feature_image = load_image(feature_filename, force_4d=True)
        if normalize:
            feature_image.get_data()[:] = self.normalization_func(feature_image.get_data())
        affine = feature_image.affine.copy()
        shape = feature_image.shape
        if self.reorder:
            affine = reorder_affine(affine, shape)
        if self.crop:
            affine, shape = crop_img(feature_image, return_affine=True, pad=True)
        if self.augment_scale_std:
            scale = np.random.normal(1, self.augment_scale_std, 3)
            affine = scale_affine(affine, shape, scale)
        if target_index is None:
            target_image = feature_image
        else:
            target_image = load_image(input[target_index], force_4d=True)
        if target_resample is None:
            target_resample = self.resample
        target_image = resample(target_image, resize_affine(affine, shape, self.window), self.window,
                                interpolation=target_resample)
        if self.additive_noise_std:
            feature_image.get_data()[:] = add_noise(feature_image.get_data(), sigma_factor=self.additive_noise_std)
        affine = resize_affine(affine, shape, self.window)
        input_image = resample(feature_image, affine, self.window, interpolation=self.resample)
        return input_image, target_image

    def get_image(self, idx, normalize=True):
        input_image, target_image = self.resample_image(self.epoch_filenames[idx], normalize=normalize)
        return input_image, target_image


class WholeBrainLabeledAutoEncoder(WholeBrainAutoEncoder):
    def __init__(self, *args, target_resample="nearest", target_index=2, labels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_resample = target_resample
        self.target_index = target_index
        self.labels = labels

    def resample_input(self, input, normalize=True):
        input_image, target_image = self.resample_image(input,
                                                        normalize=normalize,
                                                        target_resample=self.target_resample,
                                                        target_index=self.target_index)
        target_data = target_image.get_data()
        if self.labels is None:
            self.labels = np.unique(target_data)
        target_data = np.moveaxis(compile_one_hot_encoding(np.moveaxis(target_data, -1, 1),
                                                           n_labels=len(self.labels),
                                                           labels=self.labels), 1, -1)
        return input_image.get_data(), target_data
