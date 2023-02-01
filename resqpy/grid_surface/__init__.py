"""Classes for RESQML objects related to surfaces."""

__all__ = [
    'GridSkin', 'generate_untorn_surface_for_layer_interface', 'generate_torn_surface_for_layer_interface',
    'generate_torn_surface_for_x_section', 'generate_untorn_surface_for_x_section', 'point_is_within_cell',
    'create_column_face_mesh_and_surface', 'find_intersections_of_trajectory_with_surface',
    'find_intersections_of_trajectory_with_layer_interface', 'find_first_intersection_of_trajectory_with_surface',
    'find_first_intersection_of_trajectory_with_layer_interface',
    'find_first_intersection_of_trajectory_with_cell_surface',
    'find_intersection_of_trajectory_interval_with_column_face', 'trajectory_grid_overlap',
    'populate_blocked_well_from_trajectory', 'generate_surface_for_blocked_well_cells',
    'find_faces_to_represent_surface_staffa', 'find_faces_to_represent_surface_regular',
    'find_faces_to_represent_surface_regular_optimised', 'find_faces_to_represent_surface', 'bisector_from_faces',
    'column_bisector_from_faces', 'shadow_from_faces', 'get_boundary', 'where_true', 'first_true', 'intersect_numba'
]

from ._grid_skin import GridSkin
from ._grid_surface import  \
    generate_untorn_surface_for_layer_interface,  \
    generate_torn_surface_for_layer_interface,  \
    generate_torn_surface_for_x_section,  \
    generate_untorn_surface_for_x_section,  \
    point_is_within_cell,  \
    create_column_face_mesh_and_surface
from ._trajectory_intersects import  \
    find_intersections_of_trajectory_with_surface,  \
    find_intersections_of_trajectory_with_layer_interface,  \
    find_first_intersection_of_trajectory_with_surface,  \
    find_first_intersection_of_trajectory_with_layer_interface,  \
    find_first_intersection_of_trajectory_with_cell_surface,  \
    find_intersection_of_trajectory_interval_with_column_face,  \
    trajectory_grid_overlap
from ._blocked_well_populate import  \
    populate_blocked_well_from_trajectory,  \
    generate_surface_for_blocked_well_cells
from ._find_faces import  \
    find_faces_to_represent_surface_staffa,  \
    find_faces_to_represent_surface_regular,  \
    find_faces_to_represent_surface_regular_optimised,  \
    find_faces_to_represent_surface,  \
    bisector_from_faces,  \
    column_bisector_from_faces,  \
    shadow_from_faces,  \
    get_boundary,  \
    where_true,  first_true,  \
    intersect_numba

# Set "module" attribute of all public objects to this path.
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
