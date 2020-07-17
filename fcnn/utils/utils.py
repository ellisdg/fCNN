import sys

import nibabel as nib
import numpy as np
import json
from nilearn.image import resample_to_img, reorder_img, new_img_like


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


def foreground_zero_mean_normalize_image_data(data, channel_dim=4, background_value=0, tolerance=1e-5):
    data = np.copy(data)
    if data.ndim == channel_dim or data.shape[channel_dim] == 1:
        # only 1 channel, so the std and mean calculations are straight forward
        foreground_mask = np.abs(data) > (background_value + tolerance)
        foreground = data[foreground_mask]
        mean = foreground.mean()
        std = foreground.std()
        data[foreground_mask] = np.divide(foreground - mean, std)
        return data
    else:
        # std and mean need to be calculated for each channel in the 4th dimension
        for channel in range(data.shape[channel_dim]):
            channel_data = data[..., channel]
            channel_mask = np.abs(channel_data) > (background_value + tolerance)
            channel_foreground = channel_data[channel_mask]
            channel_mean = channel_foreground.mean()
            channel_std = channel_foreground.std()
            channel_data[channel_mask] = np.divide(channel_foreground - channel_mean, channel_std)
            data[..., channel] = channel_data
        return data


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


def zero_one_window(data, axis=(0, 1, 2), ceiling_percentile=99, floor_percentile=1, floor=0, ceiling=1,
                    channels_axis=None):
    data = np.copy(data)
    if len(axis) != data.ndim:
        floor_threshold = np.percentile(data, floor_percentile, axis=axis)
        if channels_axis is None:
            for i in range(data.ndim):
                if i not in axis and (i - data.ndim) not in axis:
                    # I don't understand the second part of this if statement
                    # answer: it is checking ot make sure that the axis is not indexed in reverse (i.e. axis 3 might be
                    # indexed as -1)
                    channels_axis = i
        data = np.moveaxis(data, channels_axis, 0)
        for channel in range(data.shape[0]):
            channel_data = data[channel]
            bg_mask = channel_data <= floor_threshold[channel]
            fg = channel_data[bg_mask == False]
            ceiling_threshold = np.percentile(fg, ceiling_percentile)
            channel_data = (channel_data - floor_threshold[channel])/(ceiling_threshold - floor_threshold[channel])
            channel_data[channel_data < floor] = floor
            channel_data[channel_data > ceiling] = ceiling
            data[channel] = channel_data
        data = np.moveaxis(data, 0, channels_axis)
    else:
        floor_threshold = np.percentile(data, floor_percentile)
        fg_mask = data > floor_threshold
        fg = data[fg_mask]
        ceiling_threshold = np.percentile(fg, ceiling_percentile)
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


def compile_one_hot_encoding(data, n_labels, labels=None, dtype=np.uint8, return_4d=True):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :param dtype: output type of the array
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    data = np.asarray(data)
    while len(data.shape) < 5:
        data = data[None]
    assert data.shape[1] == 1
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, dtype=dtype)
    for label_index in range(n_labels):
        if labels is not None:
            if type(labels[label_index]) == list:
                for label in labels[label_index]:
                    y[:, label_index][data[:, 0] == label] = 1
            else:
                y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    if return_4d:
        assert y.shape[0] == 1
        y = y[0]
    return y


def convert_one_hot_to_label_map(one_hot_encoding, labels, axis=1):
    label_map = np.argmax(one_hot_encoding, axis=axis)
    for index, label in enumerate(labels):
        label_map[label_map == index] = label
    return label_map


def copy_image(image):
    return image.__class__(np.copy(image.dataobj), image.affine)


def update_progress(progress, bar_length=30, message=""):
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(bar_length * progress))
    text = "\r{0}[{1}] {2:.2f}% {3}".format(message, "#" * block + "-" * (bar_length - block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def combine_images(images, axis=0, resample_unequal_affines=False, interpolation="linear"):
    base_image = images[0]
    data = list()
    max_dim = len(base_image.shape)
    for image in images:
        try:
            np.testing.assert_array_equal(image.affine, base_image.affine)
        except AssertionError as error:
            if resample_unequal_affines:
                image = resample_to_img(image, base_image, interpolation=interpolation)
            else:
                raise error
        image_data = image.get_data()
        dim = len(image.shape)
        if dim < max_dim:
            image_data = np.expand_dims(image_data, axis=axis)
        elif dim > max_dim:
            max_dim = max(max_dim, dim)
            data = [np.expand_dims(x, axis=axis) for x in data]
        data.append(image_data)
    if len(data[0].shape) > 3:
        array = np.concatenate(data, axis=axis)
    else:
        array = np.stack(data, axis=axis)
    return base_image.__class__(array, base_image.affine)


def move_channels_last(data):
    return np.moveaxis(data, 0, -1)


def move_channels_first(data):
    return np.moveaxis(data, -1, 0)


def nib_load_files(filenames, reorder=False, interpolation="linear"):
    if type(filenames) != list:
        filenames = [filenames]
    return [load_image(filename, reorder=reorder, interpolation=interpolation, force_4d=False)
            for filename in filenames]


def load_image(filename, feature_axis=3, resample_unequal_affines=True, interpolation="linear", force_4d=False,
               reorder=False):
    """
    :param feature_axis: axis along which to combine the images, if necessary.
    :param filename: can be either string path to the file or a list of paths.
    :return: image containing either the 1 image in the filename or a combined image based on multiple filenames.
    """

    if type(filename) != list:
        if not force_4d:
            return load_single_image(filename=filename, resample=interpolation, reorder=reorder)
        else:
            filename = [filename]

    return combine_images(nib_load_files(filename, reorder=reorder, interpolation=interpolation), axis=feature_axis,
                          resample_unequal_affines=resample_unequal_affines, interpolation=interpolation)


def load_single_image(filename, resample=None, reorder=True):
    image = nib.load(filename)
    if reorder:
        return reorder_img(image, resample=resample)
    return image


def extract_sub_volumes(image, sub_volume_indices):
    data = image.dataobj[..., sub_volume_indices]
    return new_img_like(ref_niimg=image, data=data)


def mask(data, threshold=0, dtype=np.float):
    return np.asarray(data > threshold, dtype=dtype)
