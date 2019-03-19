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

