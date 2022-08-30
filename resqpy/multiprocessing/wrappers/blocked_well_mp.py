"""Multiprocessing wrapper functions for the well/BlockedWell class."""

import logging

log = logging.getLogger(__name__)

from typing import Tuple, Union, List
from resqpy.model import new_model, Model
from resqpy.grid import RegularGrid
from resqpy.well import BlockedWell, Trajectory
from resqpy.multiprocessing import function_multiprocessing
from pathlib import Path
from uuid import UUID
import uuid


def blocked_well_from_trajectory_wrapper(
    index: int,
    grid_epc: str,
    grid_uuid: Union[UUID, str],
    trajectory_epc: str,
    trajectory_uuids: List[Union[UUID, str]],
) -> Tuple[int, bool, str, List[Union[UUID, str]]]:
    """Wrapper function of the BlockedWell initialisation from a Trajectory.

    Used for multiprocessing to create a new model that is saved in a temporary epc file
    and returns the required values, which are used in the multiprocessing function to
    recombine all the objects into a single epc file.

    Args:
        index (int): the index of the function call from the multiprocessing function.
        grid_epc (str): epc file path where the grid is saved.
        grid_uuid (UUID/str): UUID (universally unique identifier) of the grid object.
        trajectory_epc (str): epc file path where the trajectories are saved.
        trajectory_uuids (List[UUID/str]): a list of the trajectory uuids used to create each
            Trajectory object.

    Returns:
        Tuple containing:

            - index (int): the index passed to the function.
            - success (bool): True if all the BlockedWell objects could be created, False
              otherwise.
            - epc_file (str): the epc file path where the objects are stored.
            - uuid_list (List[UUID/str]): list of UUIDs of relevant objects.
    """
    uuid_list = []
    tmp_dir = Path("tmp_dir") / f"{uuid.uuid4()}"
    tmp_dir.mkdir(parents = True, exist_ok = True)
    epc_file = tmp_dir / "wrapper.epc"
    model = new_model(epc_file = epc_file)

    trajectory_model = Model(trajectory_epc)
    grid_model = Model(grid_epc)
    model.copy_uuid_from_other_model(grid_model, uuid = grid_uuid)

    grid = RegularGrid(grid_model, grid_uuid)

    success = True
    for trajectory_uuid in trajectory_uuids:
        model.copy_uuid_from_other_model(trajectory_model, uuid = trajectory_uuid)
        trajectory = Trajectory(
            model,
            trajectory_uuid,
        )

        blocked_well = BlockedWell(
            model,
            grid = grid,
            trajectory = trajectory,
        )
        if blocked_well is None or blocked_well.cell_count is None or blocked_well.node_count is None:
            success = False
            continue
        blocked_well.write_hdf5()
        blocked_well.create_xml()
        uuid_list.append(blocked_well.uuid)

    model.store_epc()

    return index, success, epc_file, uuid_list


def blocked_well_from_trajectory_batch(
    grid_epc: str,
    grid_uuid: Union[UUID, str],
    trajectory_epc: str,
    trajectory_uuids: List[Union[UUID, str]],
    recombined_epc: str,
    cluster,
    n_workers: int,
    require_success: bool = False,
) -> List[bool]:
    """Creates BlockedWell objects from a common RegularGrid and a list of trajectories uuids in parallel.

    Args:
        grid_epc (str): epc file path where the grid is saved.
        grid_uuid (UUID/str): UUID (universally unique identifier) of the grid object.
        trajectory_epc (str): epc file path where the trajectories are saved.
        trajectory_uuids (List[UUID/str]): a list of the trajectory uuids used to create each
            Trajectory object.
        recombined_epc (Path/str): A pathlib Path or path string of
            where the combined epc will be saved.
        cluster (LocalCluster/JobQueueCluster): a LocalCluster is a Dask cluster on a
            local machine. If using a job queing system, a JobQueueCluster can be used
            such as an SGECluster, SLURMCluster, PBSCluster, LSFCluster etc.
        n_workers (int): the number of workers on the cluster.
        require_success (bool, default False): if True an exception is raised if any failures

    Returns:
        success_list (List[bool]): A boolean list of successful function calls.
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

    success_list = function_multiprocessing(blocked_well_from_trajectory_wrapper,
                                            kwargs_list,
                                            recombined_epc,
                                            cluster,
                                            require_success = require_success)

    return success_list
