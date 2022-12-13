"""Grid Connection Set class and related functions."""

__all__ = [
    'GridConnectionSet', 'pinchout_connection_set', 'k_gap_connection_set', 'cell_set_skin_connection_set',
    'add_connection_set_and_tmults', 'grid_columns_property_from_gcs_property', 'zero_base_cell_indices_in_faces_df',
    'standardize_face_indicator_in_faces_df', 'remove_external_faces_from_faces_df', 'combined_tr_mult_from_gcs_mults'
]

from ._grid_connection_set import GridConnectionSet
from ._gcs_functions import pinchout_connection_set, k_gap_connection_set,  \
    cell_set_skin_connection_set, add_connection_set_and_tmults,  \
    grid_columns_property_from_gcs_property, zero_base_cell_indices_in_faces_df,  \
    standardize_face_indicator_in_faces_df, remove_external_faces_from_faces_df,  \
    combined_tr_mult_from_gcs_mults

# Set "module" attribute of all public objects to this path.
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
