import numpy as np
import json


def load_json(filename):
    with open(filename, 'r') as opened_file:
        return json.load(opened_file)


def dump_json(dataobj, filename):
    with open(filename, 'w') as opened_file:
        json.dump(dataobj, opened_file)


def logical_and(array_list):
    array = array_list[0]
    for other_array in array_list[1:]:
        array = np.logical_and(array, other_array)
    return array


def logical_or(array_list):
    array = array_list[0]
    for other_array in array_list[1:]:
        array = np.logical_or(array, other_array)
    return array


def get_index_value(iterable, index):
    if iterable:
        return iterable[index]


def read_polydata(filename):
    import vtk
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def extract_polydata_vertices(polydata):
    return np.asarray([polydata.GetPoint(index) for index in range(polydata.GetNumberOfPoints())])


def zero_mean_normalize_image_data(data, axis=(0, 1, 2)):
    return np.divide(data - data.mean(axis=axis), data.std(axis=axis))


def zero_floor_normalize_image_data(data, axis=(0, 1, 2), floor_percentile=1, floor=0):
    floor_threshold = np.percentile(data, floor_percentile, axis=axis)
    if data.ndim != len(axis):
        floor_threshold_shape = np.asarray(floor_threshold.shape * data.ndim)
        floor_threshold_shape[np.asarray(axis)] = 1
        floor_threshold = floor_threshold.reshape(floor_threshold_shape)
    background = data <= floor_threshold
    data = np.ma.masked_array(data - floor_threshold, mask=background)
    std = data.std(axis=axis)
    if data.ndim != len(axis):
        std = std.reshape(floor_threshold_shape)
    return np.divide(data, std).filled(floor)


def zero_one_window(data, axis=(0, 1, 2), percentile=1, floor=0, ceiling=1, channels_axis=None):
    if len(axis) != data.ndim:
        floor_threshold = np.percentile(data, percentile, axis=axis)
        if channels_axis is None:
            for i in range(data.ndim):
                if i not in axis and (i - data.ndim) not in axis:
                    channels_axis = i
        data = np.moveaxis(data, channels_axis, 0)
        for channel in range(data.shape[0]):
            channel_data = data[channel]
            bg_mask = channel_data <= floor_threshold[channel]
            fg = channel_data[bg_mask == False]
            ceiling_threshold = np.percentile(fg, 100 - percentile)
            channel_data = (channel_data - floor_threshold[channel])/(ceiling_threshold - floor_threshold[channel])
            channel_data[channel_data < floor] = floor
            channel_data[channel_data > ceiling] = ceiling
            data[channel] = channel_data
        data = np.moveaxis(data, 0, channels_axis)
    else:
        floor_threshold = np.percentile(data, percentile)
        bg_mask = data <= floor_threshold
        fg = data[bg_mask == False]
        ceiling_threshold = np.percentile(fg, 100 - percentile)
        data = (data - floor_threshold)/(ceiling_threshold - floor_threshold)
        data[data < floor] = floor
        data[data > ceiling] = ceiling
    return data


def hist_match(source, template):
    """
    Source: https://stackoverflow.com/a/33047048
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
