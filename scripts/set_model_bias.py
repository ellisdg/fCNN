from fcnn.resnet import load_model
from fcnn.utils.custom import get_metric_data_from_config
import sys
import os


def main(args):
    model_filename = os.path.abspath(args[1])
    config_filename = os.path.abspath(args[2])
    metric_filenames = [os.path.abspath(fn) for fn in args[3].split(",")]
    output_filename = os.path.abspath(args[4])
    if not os.path.exists(output_filename):
        model = load_model(model_filename)
        metric_data = get_metric_data_from_config(metric_filenames, config_filename)
        weights = model.get_weights()
        if weights[-1].shape == metric_data.shape:
            bias_index = -1
            edge_index = -2
        elif weights[-2].shape == metric_data.shape:
            bias_index = -2
            edge_index = -1
        else:
            raise ValueError("Could not find bias weights with shape {}".format(metric_data.shape))
        weights[bias_index][:] = metric_data
        weights[edge_index][:] = 0
        model.set_weights(weights)
        model.save(output_filename)


if __name__ == "__main__":
    main(sys.argv)
