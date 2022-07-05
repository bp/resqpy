"""Multiprocessing module containing the function used to run the wrapper functions in parallel."""

import logging
import time
from typing import List, Dict, Any, Callable, Union
from pathlib import Path
from resqpy.model import Model, new_model
from joblib import Parallel, delayed, parallel_backend  # type: ignore

log = logging.getLogger(__name__)


def rm_tree(path: Union[Path, str]) -> None:
    """Removes a directory using a pathlib Path.

    Args:
        path (Path/str): pathlib Path or string of the directory path.

    Returns:
        None.
    """
    path = Path(path)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    path.rmdir()


def function_multiprocessing(function: Callable,
                             kwargs_list: List[Dict[str, Any]],
                             recombined_epc: Union[Path, str],
                             cluster,
                             consolidate: bool = True,
                             require_success = False) -> List[bool]:
    """Calls a function concurrently with the specfied arguments.

    A multiprocessing pool is used to call the function multiple times in parallel. Once
    all results are returned, they are combined into a single epc file.

    Args:
        function (Callable): the wrapper function to be called. Needs to return:

            - index (int): the index of the kwargs in the kwargs_list.
            - success (bool): whether the function call was successful, whatever that
                definiton is.
            - epc_file (Path/str): the epc file path where the objects are stored.
            - uuid_list (List[str]): list of UUIDs of relevant objects.

        kwargs_list (List[Dict[Any]]): A list of keyword argument dictionaries that are
            used when calling the function.
        recombined_epc (Path/str): A pathlib Path or path string of
            where the combined epc will be saved.
        cluster (LocalCluster/JobQueueCluster): a LocalCluster is a Dask cluster on a
            local machine. If using a job queing system, a JobQueueCluster can be used
            such as an SGECluster, SLURMCluster, PBSCluster, LSFCluster etc.
        consolidate (bool): if True and an equivalent part already exists in
            a model, it is not duplicated and the uuids are noted as equivalent.
        require_success (bool): if True and any instance fails, then an exception is
            raised

    Returns:
        success_list (List[bool]): A boolean list of successful function calls.

    Note:
        This function uses the Dask backend to run the given function in parallel, so a
        Dask cluster must be setup and passed as an argument. Dask will need to be
        installed in the Python environment because it is not a dependency of the
        project. More info can be found at 
        https://resqpy.readthedocs.io/en/latest/tutorial/multiprocessing.html
    """
    log.info("multiprocessing function called with %s function.", function.__name__)

    for i, kwargs in enumerate(kwargs_list):
        kwargs["index"] = i

    with parallel_backend("dask"):
        results = Parallel()(delayed(function)(**kwargs) for kwargs in kwargs_list)

    # Sorting the results by the original kwargs_list index.
    results = list(sorted(results, key = lambda x: x[0]))

    success_list = [result[1] for result in results]
    epc_list = [result[2] for result in results]
    uuids_list = [result[3] for result in results]
    success_count = sum(success_list)
    log.info("multiprocessing function calls complete; successes: %s/%s.", success_count, len(results))
    if require_success and success_count < len(results):
        raise Exception('one or more multiprocessing instances failed')

    epc_file = Path(str(recombined_epc))
    if epc_file.is_file():
        model_recombined = Model(epc_file = str(epc_file))
    else:
        model_recombined = new_model(epc_file = str(epc_file))

    log.info("creating the recombined epc file")
    for i, epc in enumerate(epc_list):
        if epc is None:
            continue
        while True:
            try:
                model = Model(epc_file = epc)
                break
            except FileNotFoundError:
                time.sleep(1)
                continue
        uuids = uuids_list[i]
        if uuids is None:
            uuids = model.uuids()
        for uuid in uuids:
            model_recombined.copy_uuid_from_other_model(model, uuid = uuid, consolidate = consolidate)

    # Deleting temporary directory.
    log.debug("deleting the temporary directory")
    rm_tree("tmp_dir")

    model_recombined.store_epc()
    model.h5_release()

    log.info("recombined epc file complete")

    return success_list
