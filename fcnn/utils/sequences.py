import numpy as np
import nibabel as nib
from keras.utils import Sequence

from .radiomic_utils import binary_classification, multilabel_classification, fetch_data, pick_random_list_elements, \
    load_image, fetch_data_for_point
from .hcp import nib_load_files, extract_gifti_array, extract_gifti_surface_vertices, get_axis, get_vertices_from_scalar, extract_scalar_map
from .utils import read_polydata, extract_polydata_vertices


class SingleSiteSequence(Sequence):
    def __init__(self, filenames, batch_size,
                 target_labels, window, spacing, classification='binary',
                 points_per_subject=1, flip=False, reorder=False):
        self.batch_size = batch_size
        self.filenames = filenames
        self.target_labels = target_labels
        self.window = window
        self.points_per_subject = points_per_subject
        self.flip = flip
        self.reorder = reorder
        self.spacing = spacing
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
        return len(self.filenames)

    def generate_epoch_filenames(self):
        epoch_filenames = list(self.filenames)
        np.random.shuffle(epoch_filenames)
        self.epoch_filenames = list(epoch_filenames)

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
        super().__init__(filenames, batch_size, target_labels=tuple(), window=window, spacing=spacing,
                         classification=classification, **kwargs)
        self.metric_names = metric_names
        self.surface_names = surface_names

    def __getitem__(self, idx):
        batch_filenames = self.epoch_filenames[idx * self.subjects_per_batch:(idx + 1) * self.subjects_per_batch]
        batch_x = list()
        batch_y = list()
        for feature_filename, surface_filenames, metric_filenames, subject_id in batch_filenames:
            metrics = nib_load_files(metric_filenames)
            all_metric_data = list()
            for metric, metric_names in zip(metrics, self.metric_names):
                for metric_name in metric_names:
                    metric_data = list()
                    for surface_name in self.surface_names:
                        metric_data.extend(extract_scalar_map(metric, metric_name.format(subject_id),
                                                              brain_structure_name=surface_name))
                    all_metric_data.append(metric_data)
            all_metric_data = np.stack(all_metric_data, axis=1)

            # extract the vertices
            surfaces = nib_load_files(surface_filenames)
            vertices = list()
            for surface, surface_name in zip(surfaces, self.surface_names):
                vertices_index = get_vertices_from_scalar(metrics[0], brain_structure_name=surface_name)
                surface_vertices = extract_gifti_surface_vertices(surface, primary_anatomical_structure=surface_name)
                vertices.extend(surface_vertices[vertices_index])
            vertices = np.asarray(vertices)

            # randomly select the target vertices and corresponding values
            indices = np.random.choice(np.arange(vertices.shape[0]), size=self.points_per_subject, replace=False)
            random_vertices = vertices[indices]
            random_target_values = all_metric_data[indices]

            # load data
            feature_image = load_image(feature_filename, reorder=self.reorder)
            for vertex, y in zip(random_vertices, random_target_values):
                x = fetch_data_for_point(vertex, feature_image, window=self.window, flip=self.flip,
                                         spacing=self.spacing)
                batch_x.append(x)
                batch_y.append(y)
        return np.asarray(batch_x), np.asarray(batch_y)


class SubjectPredictionSequence(Sequence):
    def __init__(self, feature_filename, surface_filename, surface_name, batch_size=50, window=(64, 64, 64), flip=False,
                 spacing=(1, 1, 1)):
        self.feature_image = nib.load(feature_filename)
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
