"""Multiprocessing wrapper functions for the Mesh class."""

import logging

log = logging.getLogger(__name__)

import uuid
from typing import Tuple, Union, List
from pathlib import Path
from uuid import UUID

import resqpy.model as rq
import resqpy.grid as grr
import resqpy.surface as rqs
import resqpy.multi_processing as rqmp


def mesh_from_regular_grid_column_property_wrapper(
    index: int,
    parent_tmp_dir: str,
    grid_epc: str,
    grid_uuid: Union[UUID, str],
    prop_uuids: List[Union[UUID, str]],
) -> Tuple[int, bool, str, List[Union[UUID, str]]]:
    """Multiprocessing wrapper function of the Mesh from_regular_grid_column_property method.

    arguments:
        index (int): the index of the function call from the multiprocessing function
        parent_tmp_dir (str): the parent temporary directory path from the multiprocessing function
        grid_epc (str): epc file path where the grid is saved
        grid_uuid (UUID or str): UUID (universally unique identifier) of the regular grid object
        prop_uuids (list of UUID or str): a list of the property uuids used to create each Mesh
            and their relationship

    returns:
        Tuple containing:
            - index (int): the index passed to the function
            - success (bool): True if all the Mesh objects could be created, False otherwise
            - epc_file (str): the epc file path where the objects are stored
            - uuid_list (List[UUID/str]): list of UUIDs of relevant objects

    notes:
        Use this function as argument to multiprocessing function; it will create a new model that
        is saved in a temporary epc file and returns the required values, which are used in the
        multiprocessing function to recombine all the objects into a single epc file; applications
        should generally call mesh_from_regular_grid_column_property_batch() which makes use of
        this wrapper
    """
    uuid_list = []
    tmp_dir = Path(parent_tmp_dir) / f"{uuid.uuid4()}"
    tmp_dir.mkdir(parents = True, exist_ok = True)
    epc_file = str(tmp_dir / "wrapper.epc")
    model = rq.new_model(epc_file = epc_file, quiet = True)

    g_model = rq.Model(grid_epc, quiet = True)
    g_crs_uuid = g_model.uuid(obj_type = "LocalDepth3dCrs",
                              related_uuid = grid_uuid)  # todo: check this relationship exists

    if g_crs_uuid is not None:
        model.copy_uuid_from_other_model(g_model, g_crs_uuid)
        uuid_list.append(g_crs_uuid)
    model.copy_uuid_from_other_model(g_model, uuid = grid_uuid)

    for prop_uuid in prop_uuids:
        model.copy_uuid_from_other_model(g_model, uuid = prop_uuid)

    grid = grr.RegularGrid(parent_model = model, uuid = grid_uuid)

    success = True
    for prop_uuid in prop_uuids:
        mesh = rqs.Mesh.from_regular_grid_column_property(model, grid.uuid, prop_uuid)
        if mesh is None:
            success = False
            continue
        mesh.write_hdf5()
        mesh.create_xml()
        uuid_list.append(mesh.uuid)
        model.create_reciprocal_relationship_uuids(mesh.uuid, "sourceObject", prop_uuid, "detinationObject")

    model.store_epc(quiet = True)

    return index, success, epc_file, uuid_list


def mesh_from_regular_grid_column_property_batch(grid_epc: str,
                                                 grid_uuid: Union[UUID, str],
                                                 prop_uuids: List[Union[UUID, str]],
                                                 recombined_epc: str,
                                                 cluster,
                                                 n_workers: int,
                                                 require_success: bool = False,
                                                 tmp_dir_path: Union[Path, str] = '.') -> List[bool]:
    """Creates Mesh objects from a list of property uuids in parallel.

    arguments:
        grid_epc (str): epc file path where the grid is saved
        grid_uuid (UUID/str): UUID (universally unique identifier) of the grid object
        prop_uuids (List[UUID/str]): a list of the column property uuids used to create each Mesh
            and their relationship
        recombined_epc (Path/str): A pathlib Path or path string of
            where the combined epc will be saved
        cluster (LocalCluster/JobQueueCluster): a LocalCluster is a Dask cluster on a
            local machine. If using a job queing system, a JobQueueCluster can be used
            such as an SGECluster, SLURMCluster, PBSCluster, LSFCluster etc
        n_workers (int): the number of workers on the cluster
        require_success (bool, default False): if True an exception is raised if any failures
        tmp_dir_path (str or Path, default '.'): the directory within which temporary directories will reside

    returns:
        success_list (list of bool): A boolean list of successful function calls; one value per batch
    """
    n_uuids = len(prop_uuids)
    prop_uuids_list = [prop_uuids[i * n_uuids // n_workers:(i + 1) * n_uuids // n_workers] for i in range(n_workers)]

    kwargs_list = []
    for prop_uuids in prop_uuids_list:
        d = {
            "grid_epc": grid_epc,
            "grid_uuid": grid_uuid,
            "prop_uuids": prop_uuids,
        }
        kwargs_list.append(d)

    success_list = rqmp.function_multiprocessing(mesh_from_regular_grid_column_property_wrapper,
                                                 kwargs_list,
                                                 recombined_epc,
                                                 cluster,
                                                 require_success = require_success,
                                                 tmp_dir_path = tmp_dir_path)

    return success_list
