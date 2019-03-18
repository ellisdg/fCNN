import numpy as np
from keras.utils import Sequence

from .radiomic_utils import binary_classification, multilabel_classification, fetch_data, pick_random_list_elements, \
    load_image, fetch_data_for_point
from .hcp import nib_load_files, extract_gifti_array, extract_gifti_surface_vertices


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
    def __init__(self, filenames, batch_size, target_labels, window, spacing, metric_names, classification=None,
                 surface_names=('CortexLeft', 'CortexRight'), **kwargs):
        super().__init__(filenames, batch_size, target_labels, window, spacing, classification=classification, **kwargs)
        self.metric_names = metric_names
        self.surface_names = surface_names

    def __getitem__(self, idx):
        batch_filenames = self.epoch_filenames[idx * self.subjects_per_batch:(idx + 1) * self.subjects_per_batch]
        batch_x = list()
        batch_y = list()
        for feature_filename, surface_filenames, metric_filenames, subject_id in batch_filenames:
            # extract the vertices
            surfaces = nib_load_files(surface_filenames)
            # vertices should be shape (n_vertices, 3)
            # enforce correct left/right ordering of vertex data
            vertices = np.concatenate([extract_gifti_surface_vertices(surface,
                                                                      anatomical_structure_primary=surface_name)
                                       for surface, surface_name in zip(surfaces, self.surface_names)],
                                      axis=0)

            # extract the target values corresponding to the vertices
            target_data = list()
            metrics = nib_load_files(metric_filenames)
            for surface_name, metric in zip(self.surface_names, metrics):
                # enforce correct left/right ordering of metric data
                assert metric.meta.meta['AnatomicalStructurePrimary'] == surface_name
                hemisphere_metric_data = list()
                for metric_name in self.metric_names:
                    subject_metric_name = metric_name.format(subject_id)
                    array = extract_gifti_array(gifti_object=metric, index=subject_metric_name)
                    hemisphere_metric_data.append(array)
                hemisphere_metric_data = np.stack(hemisphere_metric_data, axis=1)
                target_data.append(hemisphere_metric_data)
            target_data = np.concatenate(target_data, axis=0)  # should be shape (n_vertices, n_metrics)

            # randomly select the target vertices and corresponding values
            indices = np.random.choice(np.arange(vertices.shape[0]), size=self.points_per_subject, replace=False)
            random_vertices = vertices[indices]
            random_target_values = target_data[indices]

            # load data
            feature_image = load_image(feature_filename, reorder=self.reorder)
            for vertex, y in zip(random_vertices, random_target_values):
                x = fetch_data_for_point(vertex, feature_image, window=self.window, flip=self.flip,
                                         spacing=self.spacing)
                batch_x.append(x)
                batch_y.append(y)
        return np.asarray(batch_x), np.asarray(batch_y)
