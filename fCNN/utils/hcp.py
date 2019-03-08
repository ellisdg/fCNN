import nibabel as nib


def nib_load_files(filenames):
    return (nib.load(filename) for filename in filenames)


def get_vertex_data_from_surface(surface,
                                 index=0,
                                 geometric_type="Anatomical",
                                 primary_anatomical_structure=None,
                                 secondary_anatomical_structure=None):
    array = surface.darrays[index]
    if geometric_type:
        assert array.metadata["GeometricType"] == geometric_type
    if primary_anatomical_structure:
        assert array.metadata["AnatomicalStructurePrimary"] == primary_anatomical_structure
    if secondary_anatomical_structure:
        assert array.metadata["AnatomicalStructureSecondary"] == secondary_anatomical_structure
    return array.data
