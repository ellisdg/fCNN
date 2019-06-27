from torch.utils.data import Dataset
import torch

from ..sequences import WholeBrainRegressionSequence, nib_load_files, get_metric_data


class WholeBrainCIFTI2DenseScalarDataset(WholeBrainRegressionSequence, Dataset):
    def __init__(self, *args, batch_size=1, **kwargs):
        super().__init__(*args, batch_size=batch_size, **kwargs)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        feature_filename, surface_filenames, metric_filenames, subject_id = self.filenames[idx]
        metrics = nib_load_files(metric_filenames)
        x = self.resample_input(feature_filename)
        y = get_metric_data(metrics, self.metric_names, self.surface_names, subject_id).T.ravel()
        return torch.from_numpy(x), torch.from_numpy(y)
