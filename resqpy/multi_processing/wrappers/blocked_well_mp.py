"""Multiprocessing wrapper functions for the well/BlockedWell class."""

import logging

log = logging.getLogger(__name__)

import uuid
from typing import Tuple, Union, List
from pathlib import Path
from uuid import UUID

import resqpy.model as rq
import resqpy.grid as grr
import resqpy.well as rqw
import resqpy.multi_processing as rqmp


def blocked_well_from_trajectory_wrapper(
    index: int,
    parent_tmp_dir: str,
    grid_epc: str,
    grid_uuid: Union[UUID, str],
    trajectory_epc: str,
    trajectory_uuids: List[Union[UUID, str]],
) -> Tuple[int, bool, str, List[Union[UUID, str]]]:
    """Multiprocessing wrapper function of the BlockedWell initialisation from a Trajectory.

    arguments:
        index (int): the index of the function call from the multiprocessing function
        parent_tmp_dir (str): the parent temporary directory path from the multiprocessing function
        grid_epc (str): epc file path where the grid is saved
        grid_uuid (UUID or str): UUID (universally unique identifier) of the grid object
        trajectory_epc (str): epc file path where the trajectories are saved
        trajectory_uuids (list of UUID or str): a list of the trajectory uuids used to create each
            Trajectory object

    returns:
        Tuple containing:
        - index (int): the index passed to the function;
        - success (bool): True if all the BlockedWell objects could be created, False otherwise;
        - epc_file (str): the epc file path where the objects are stored;
        - uuid_list (List[UUID/str]): list of UUIDs of relevant objects

    note:
        used this wrapper when calling the multiprocessing function to initialise blocked wells from
        trajectories; it will create a new model that is saved in a temporary epc file
        and returns the required values, which are used in the multiprocessing function to
        recombine all the objects into a single epc file
    """
    uuid_list = []
    tmp_dir = Path(parent_tmp_dir) / f"{uuid.uuid4()}"
    tmp_dir.mkdir(parents = True, exist_ok = True)
    epc_file = str(tmp_dir / "wrapper.epc")
    model = rq.new_model(epc_file = epc_file, quiet = True)

    trajectory_model = rq.Model(trajectory_epc, quiet = True)
    grid_model = rq.Model(grid_epc)
    model.copy_uuid_from_other_model(grid_model, uuid = grid_uuid)

    grid = grr.any_grid(grid_model, uuid = grid_uuid)

    success = True
    for trajectory_uuid in trajectory_uuids:
        model.copy_uuid_from_other_model(trajectory_model, uuid = trajectory_uuid)
        trajectory = rqw.Trajectory(model, trajectory_uuid)
        blocked_well = rqw.BlockedWell(model, grid = grid, trajectory = trajectory)
        if blocked_well is None or blocked_well.cell_count is None or blocked_well.node_count is None:
            success = False
            continue
        blocked_well.write_hdf5()
        blocked_well.create_xml()
        uuid_list.append(blocked_well.uuid)

    model.store_epc(quiet = True)

    return index, success, epc_file, uuid_list


def blocked_well_from_trajectory_batch(grid_epc: str,
                                       grid_uuid: Union[UUID, str],
                                       trajectory_epc: str,
                                       trajectory_uuids: List[Union[UUID, str]],
                                       recombined_epc: str,
                                       cluster,
                                       n_workers: int,
                                       require_success: bool = False,
                                       tmp_dir_path: Union[Path, str] = '.') -> List[bool]:
    """Creates BlockedWell objects from a common grid and a list of trajectories' uuids, in parallel.

    arguments:
        grid_epc (str): epc file path where the grid is saved
        grid_uuid (UUID or str): UUID (universally unique identifier) of the grid object
        trajectory_epc (str): epc file path where the trajectories are saved
        trajectory_uuids (list of UUID or str): a list of the trajectory uuids used to create each
            Trajectory object
        recombined_epc (Path or str): A pathlib Path or path string, where the combined epc will be saved
        cluster (LocalCluster/JobQueueCluster): a LocalCluster is a Dask cluster on a
            local machine; if using a job queing system, a JobQueueCluster can be used
            such as an SGECluster, SLURMCluster, PBSCluster, LSFCluster etc
        n_workers (int): the number of workers on the cluster
        require_success (bool, default False): if True an exception is raised if any failures
        tmp_dir_path (str or Path, default '.'): the directory within which temporary directories will reside

    returns:
        success_list (list of bool): A boolean list of successful function calls

    notes:
        the returned success list contains one value per batch, set True if all blocked wells
        were successfully created in the batch, False if one or more failed in the batch
    """
    n_uuids = len(trajectory_uuids)
    trajectory_uuids_list = [
        trajectory_uuids[i * n_uuids // n_workers:(i + 1) * n_uuids // n_workers] for i in range(n_workers)
    ]

    kwargs_list = []
    for trajectory_uuids in trajectory_uuids_list:
        d = {
            "grid_epc": grid_epc,
            "grid_uuid": grid_uuid,
            "trajectory_epc": trajectory_epc,
            "trajectory_uuids": trajectory_uuids,
        }
        kwargs_list.append(d)

    success_list = rqmp.function_multiprocessing(blocked_well_from_trajectory_wrapper,
                                                 kwargs_list,
                                                 recombined_epc,
                                                 cluster,
                                                 require_success = require_success,
                                                 tmp_dir_path = tmp_dir_path)

    return success_list
