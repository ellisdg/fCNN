from torch.utils.data import Dataset
import torch
import numpy as np


from ..sequences import (WholeBrainRegressionSequence, HCPRegressionSequence, nib_load_files, get_metric_data,
                         WholeBrainAutoEncoder, WholeBrainLabeledAutoEncoder, WindowedAutoEncoder,
                         SubjectPredictionSequence, fetch_data_for_point, WholeVolumeCiftiSupervisedRegressionSequence,
                         WholeVolumeSupervisedRegressionSequence)


class WholeBrainCIFTI2DenseScalarDataset(WholeBrainRegressionSequence, Dataset):
    def __init__(self, *args, batch_size=1, shuffle=False, **kwargs):
        super().__init__(*args, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def __len__(self):
        return len(self.epoch_filenames)

    def __getitem__(self, idx):
        feature_filename, surface_filenames, metric_filenames, subject_id = self.epoch_filenames[idx]
        metrics = nib_load_files(metric_filenames)
        x = self.resample_input(feature_filename)
        y = self.get_metric_data(metrics, subject_id)
        return torch.from_numpy(x).float().permute(3, 0, 1, 2), torch.from_numpy(y).float()

    def get_metric_data(self, metrics, subject_id):
        return get_metric_data(metrics, self.metric_names, self.surface_names, subject_id).T.ravel()


class HCPRegressionDataset(HCPRegressionSequence, Dataset):
    def __init__(self, *args, points_per_subject=1, **kwargs):
        super().__init__(*args, batch_size=points_per_subject, **kwargs)

    def __len__(self):
        return len(self.epoch_filenames)

    def __getitem__(self, idx):
        x, y = self.fetch_hcp_subject_batch(*self.epoch_filenames[idx])
        return torch.from_numpy(np.moveaxis(np.asarray(x), -1, 1)).float(), torch.from_numpy(np.asarray(y)).float()


class HCPSubjectDataset(SubjectPredictionSequence):
    def __init__(self, *args, batch_size=None, **kwargs):
        if batch_size is not None:
            print("Ignoring the set batch_size")
        super().__init__(*args, batch_size=1, **kwargs)

    def __getitem__(self, idx):
        x = self.fetch_data_for_index(idx)
        return torch.from_numpy(np.moveaxis(np.asarray(x), -1, 0)).float()

    def __len__(self):
        return len(self.vertices)

    def fetch_data_for_index(self, idx):
        return fetch_data_for_point(self.vertices[idx], self.feature_image, window=self.window, flip=self.flip,
                                    spacing=self.spacing)


class AEDataset(WholeBrainAutoEncoder, Dataset):
    def __init__(self, *args, batch_size=1, shuffle=False, metric_names=None, **kwargs):
        super().__init__(*args, batch_size=batch_size, shuffle=shuffle, metric_names=metric_names, **kwargs)

    def __len__(self):
        return len(self.epoch_filenames)

    def __getitem__(self, idx):
        item = self.epoch_filenames[idx]
        x, y = self.resample_input(item)
        return (torch.from_numpy(np.moveaxis(np.asarray(x), -1, 0)).float(),
                torch.from_numpy(np.moveaxis(np.asarray(y), -1, 0)).float())


class LabeledAEDataset(WholeBrainLabeledAutoEncoder, Dataset):
    def __init__(self, *args, batch_size=1, shuffle=False, metric_names=None, **kwargs):
        super().__init__(*args, batch_size=batch_size, shuffle=shuffle, metric_names=metric_names, **kwargs)

    def __len__(self):
        return len(self.epoch_filenames)

    def __getitem__(self, idx):
        item = self.epoch_filenames[idx]
        x, y = self.resample_input(item)
        return (torch.from_numpy(np.moveaxis(np.asarray(x), -1, 0)).float(),
                torch.from_numpy(np.moveaxis(np.asarray(y), -1, 0)).byte())


class WholeVolumeSupervisedRegressionDataset(WholeVolumeSupervisedRegressionSequence, Dataset):
    def __init__(self, *args, batch_size=1, shuffle=False, **kwargs):
        super().__init__(*args, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def __len__(self):
        return len(self.epoch_filenames)

    def __getitem__(self, idx):
        item = self.epoch_filenames[idx]
        x, y = self.resample_input(item)
        return (torch.from_numpy(np.moveaxis(np.asarray(x), -1, 0)).float(),
                torch.from_numpy(np.moveaxis(np.asarray(y), -1, 0)).float())


class WholeVolumeCiftiSupervisedRegressionDataset(WholeVolumeCiftiSupervisedRegressionSequence,
                                                  WholeVolumeSupervisedRegressionDataset):
    pass


class WindowedAEDataset(WindowedAutoEncoder, Dataset):
    def __init__(self, *args, points_per_subject=1, **kwargs):
        super().__init__(*args, batch_size=points_per_subject, **kwargs)

    def __len__(self):
        return len(self.epoch_filenames)

    def __getitem__(self, idx):
        x, y = self.fetch_hcp_subject_batch(*self.epoch_filenames[idx])
        return (torch.from_numpy(np.moveaxis(np.asarray(x), -1, 1)).float(),
                torch.from_numpy(np.moveaxis(np.asarray(y), -1, 1)).float())
