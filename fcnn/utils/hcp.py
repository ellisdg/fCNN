import nibabel as nib
import numpy as np


def nib_load_files(filenames):
    return [nib.load(filename) for filename in filenames]


def extract_gifti_surface_vertices(surface, index=0, geometric_type="Anatomical", **kwargs):
    return extract_gifti_array(surface, index=index, geometric_type=geometric_type, **kwargs)


def extract_gifti_array(gifti_object,
                        index,
                        geometric_type=None,
                        primary_anatomical_structure=None,
                        secondary_anatomical_structure=None):
    if type(index) is str:
        index = extract_gifti_array_names(gifti_object).index(index)
    array = gifti_object.darrays[index]
    if geometric_type:
        assert array.metadata["GeometricType"] == geometric_type
    if primary_anatomical_structure:
        assert array.metadata["AnatomicalStructurePrimary"] == primary_anatomical_structure
    if secondary_anatomical_structure:
        assert array.metadata["AnatomicalStructureSecondary"] == secondary_anatomical_structure
    return array.data


def extract_gifti_array_names(gifti_object, key='Name'):
    return [array.metadata[key] for array in gifti_object.darrays]


def extract_scalar_map_names(pscalar, map_index=0):
    return [index.map_name for index in pscalar.header.get_index_map(map_index)]


def extract_scalar_map(pscalar, map_name, brain_structure_name=None, brain_model_axis_index=1):
    map_names = extract_scalar_map_names(pscalar)
    data = pscalar.dataobj[map_names.index(map_name)]
    if brain_structure_name is not None:
        data = data[get_mask_from_scalar(pscalar,
                                         brain_structure_name=brain_structure_name,
                                         axis_index=brain_model_axis_index)]
    return data


def extract_parcellated_scalar_parcel_names(pscalar, parcel_index=1):
    parcel_names = list()
    for index in pscalar.header.get_index_map(parcel_index):
        try:
            parcel_names.append(index.name)
        except AttributeError:
            continue
    if not pscalar.shape[parcel_index] == len(parcel_names):
        raise RuntimeError("Number of parcel names, {}, does not match pscalar shape, {}.".format(len(parcel_names),
                                                                                                  pscalar.shape))
    return parcel_names


def get_vertices_from_scalar(scalar, brain_structure_name, axis_index=1):
    brain_model_axis = get_axis(scalar, axis_index)
    return brain_model_axis.vertex[get_mask_from_axis(brain_model_axis, brain_structure_name)]


def get_mask_from_scalar(scalar, brain_structure_name, axis_index=1):
    return get_mask_from_axis(scalar.header.get_axis(axis_index), brain_structure_name)


def get_mask_from_axis(brain_model_axis, brain_structure_name):
    return brain_model_axis.name == brain_model_axis.to_cifti_brain_structure_name(brain_structure_name)


def get_axis(scalar, axis_index):
    return scalar.header.get_axis(axis_index)


def get_metric_data(metrics, metric_names, surface_names, subject_id, stack_axis=1):
    all_metric_data = list()
    for metric, metric_names in zip(metrics, metric_names):
        for metric_name in metric_names:
            metric_data = list()
            for surface_name in surface_names:
                metric_data.extend(extract_scalar_map(metric, metric_name.format(subject_id),
                                                      brain_structure_name=surface_name))
            all_metric_data.append(metric_data)
    return np.stack(all_metric_data, axis=stack_axis)


def new_cifti_scalar_like(array, scalar_names, structure_names, reference_cifti, default_value=0):
    scalar_axis = reference_cifti.header.get_axis(0)
    new_scalar_axis = scalar_axis.__class__(scalar_names)
    model_axis = reference_cifti.header.get_axis(1)
    dataobj = np.ones((array.shape[0], model_axis.name.shape[0]), array.dtype) * default_value
    i = 0
    for structure_name in structure_names:
        structure_mask = model_axis.name == model_axis.to_cifti_brain_structure_name(structure_name)
        ii = np.sum(structure_mask) + i
        dataobj[:, structure_mask] = array[:, i:ii]
        i = ii
    if default_value == 0:
        np.testing.assert_almost_equal(np.sum(dataobj), np.sum(array))
    return reference_cifti.__class__(dataobj=dataobj, header=[new_scalar_axis, model_axis])
