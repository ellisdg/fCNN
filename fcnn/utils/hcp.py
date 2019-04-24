import nibabel as nib


def nib_load_files(filenames):
    return (nib.load(filename) for filename in filenames)


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


def extract_scalar_map(pscalar, map_name):
    map_names = extract_scalar_map_names(pscalar)
    return pscalar.dataobj[map_names.index(map_name)]


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
