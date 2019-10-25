from torch.utils.data import Dataset
import torch
import numpy as np


from ..sequences import WholeBrainRegressionSequence, HCPRegressionSequence, nib_load_files, get_metric_data


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
    def __init__(self, *args, batch_size=1, **kwargs):
        super().__init__(self, *args, batch_size=batch_size, **kwargs)

    def __len__(self):
        return len(self.epoch_filenames)

    def __getitem__(self, idx):
        x, y = self.fetch_hcp_subject_batch(*self.epoch_filenames[idx])
        return np.asarray(x).swapaxes(-1, 1), np.asarray(y)
