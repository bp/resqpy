from typing import Tuple, Union, List
import resqpy.grid_surface as rqgs
from resqpy.model import new_model
from resqpy.grid import RegularGrid
from resqpy.surface import Surface
from resqpy.property import PropertyCollection


def find_faces_to_represent_surface_regular_wrapper(grid: RegularGrid, surface: Surface, name: str, tmp_dir: str,
                                                    index: int) -> Tuple[Union[bool, str, List[str], int]]:
    """Wrapper function of find_faces_to_represent_surface_regular_optimised.
    
    Used for multiprocessing to create a new model that is saved in a temporary epc file
    and returns the required values, which are used in the multiprocessing function to
    recombine all the objects into a single epc file.
    
    Args:
        grid (RegularGrid): the grid for which to create a grid connection set
            representation of the surface.
        surface (Surface): the surface to be intersected with the grid.
        name (str): the feature name to use in the grid connection set.
        tmp_dir (str): path of the temporary directory that will hold the epc file for
            relevant objects that are saved in this function.
        index (int): the index of the function call from the multiprocessing function.

    Returns:
        Tuple containing:

            - success (bool): whether the function call was successful, whatever that
                definiton is.
            - epc_file (str): the epc file path where the objects are stored.
            - uuid_list (List[str]): list of UUIDs of relevant objects.
            - index (int): the index passed to the function.
    """

    epc_file = f"{tmp_dir}/wrapper.epc"
    model = new_model(epc_file = epc_file)
    model.copy_uuid_from_other_model(grid.model, uuid = grid.uuid)
    model.copy_uuid_from_other_model(surface.model, uuid = surface.uuid)

    uuid_list = []
    uuid_list.append(surface.uuid)

    returns = rqgs.find_faces_to_represent_surface_regular_optimised(grid, surface, name)

    if isinstance(returns, tuple):
        gcs = returns[0]
        properties = returns[1]
        property_collection = PropertyCollection(support = gcs)
        for name, array in properties.items():
            if name == "normal vector":
                property_collection.add_cached_array_to_imported_list(array, indexable_element = "faces", points = True)
            else:
                property_collection.add_cached_array_to_imported_list(array, indexable_element = "faces")
    else:
        gcs = returns

    success = False
    if gcs.count > 0:
        success = True

    model.copy_uuid_from_other_model(gcs.model, uuid = gcs.uuid)
    uuid_list.append(gcs.uuid)

    model.store_epc()

    return success, epc_file, uuid_list, index
