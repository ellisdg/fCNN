import numpy as np
from keras.utils import Sequence

from fCNN.utils.radiomic_utils import binary_classification, multilabel_classification, fetch_data, \
    pick_random_list_elements
from fCNN.utils.hcp import nib_load_files, get_vertex_data_from_surface


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
    def __init__(self, filenames, batch_size, target_labels, window, spacing, metric_names, classification=None, **kwargs):
        super().__init__(filenames, batch_size, target_labels, window, spacing, classification=classification, **kwargs)
        self.metric_names = metric_names

    def __getitem__(self, idx):
        batch_filenames = self.epoch_filenames[idx * self.subjects_per_batch:(idx + 1) * self.subjects_per_batch]
        batch_x = list()
        batch_y = list()
        for feature_filename, surface_filenames, metric_filenames in batch_filenames:
            surfaces = nib_load_files(surface_filenames)
            vertices = [get_vertex_data_from_surface(surface) for surface in surfaces]
            metrics = nib_load_files(metric_filenames)
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
