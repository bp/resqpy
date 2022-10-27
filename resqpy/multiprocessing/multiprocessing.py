"""Multiprocessing module containing the function used to run the wrapper functions in parallel."""

import logging
import os
import time
import uuid
from typing import List, Dict, Any, Callable, Union
from pathlib import Path
from joblib import Parallel, delayed, parallel_backend  # type: ignore
import resqpy.model as rq

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


def function_multiprocessing(
    function: Callable,
    kwargs_list: List[Dict[str, Any]],
    recombined_epc: Union[Path, str],
    cluster,
    consolidate: bool = True,
    require_success = False,
) -> List[bool]:
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
    log.info("multiprocessing function called with %s function, %s entries.", function.__name__, len(kwargs_list))

    tmp_dir = f'tmp_{uuid.uuid4()}'
    for i, kwargs in enumerate(kwargs_list):
        kwargs["index"] = i
        kwargs["parent_tmp_dir"] = tmp_dir

    if cluster is None:

        results = []
        for i, kwargs in enumerate(kwargs_list):
            log.debug(f'calling function for entry {i}; name: {kwargs.get("name")}')
            one_r = function(**kwargs)
            results.append(one_r)
            log.debug(f'completed entry: {one_r[0]}; success: {one_r[1]}; epc: {one_r[2]}')
            log.debug(f'uuid list: {one_r[3]}')

    else:

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
        model_recombined = rq.Model(epc_file = str(epc_file), quiet = True)
        log.info(f"updating the recombined epc file: {epc_file}")
    else:
        model_recombined = rq.new_model(epc_file = str(epc_file))
        log.info(f"creating the recombined epc file: {epc_file}")

    for i, epc in enumerate(epc_list):
        log.debug(f'recombining from mp instance {i} epc: {epc}')
        if epc is None:
            continue
        attempt = 0
        while not os.path.exists(epc):
            attempt += 1
            if attempt == 7:
                log.warning(f'mp epc slow to materialise: {epc}')
            if attempt > 300:
                raise FileNotFoundError(f'timeout waiting for multiprocess worker epc to become available: {epc}')
            time.sleep(min(attempt, 10))
        attempt = 0
        while True:
            attempt += 1
            try:
                model = rq.Model(epc_file = epc, quiet = True)
                break
            except FileNotFoundError:
                if attempt >= 10:
                    raise FileNotFoundError(f'timeout waiting for mp epc {epc}')
                time.sleep(1)
        uuids = uuids_list[i]
        if uuids is None:
            uuids = model.uuids()
        for u in uuids:
            attempt = 0
            while True:
                attempt += 1
                try:
                    model_recombined.copy_uuid_from_other_model(model, uuid = u, consolidate = consolidate)
                    break
                except BlockingIOError:
                    if attempt >= 5:
                        raise
                time.sleep(1)

    # Deleting temporary directory.
    log.debug(f"deleting the temporary directory {tmp_dir}")
    rm_tree(tmp_dir)

    model_recombined.store_epc(quiet = True)
    model.h5_release()

    log.debug(f"recombined epc file complete: {epc_file}")

    return success_list
