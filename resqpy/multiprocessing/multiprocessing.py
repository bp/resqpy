from multiprocessing import Pool
from typing import List, Optional, Dict, Any, Callable, Union
from pathlib import Path
import logging
import tempfile
from resqpy.model import Model, new_model

log = logging.getLogger(__name__)


def rm_tree(path: Path) -> None:
    """Removes a directory using a pathlib Path.
    
    Args:
        path (Path): pathlib Path of the directory.

    Returns:
        None.
    """
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    path.rmdir()


def function_multiprocessing(function: Callable,
                             kwargs_list: List[Dict[str, Any]],
                             recombined_epc: Optional[Union[Path, str]] = None,
                             consolidate: bool = True,
                             processes: Optional[int] = None) -> List[bool]:
    """Calls a function concurrently with the specfied arguments.
    
    A multiprocessing pool is used to call the function multiple times in parallel. Once
    all results are returned, they are combined into a single epc file.

    Args:
        function (Callable): the function to be called. Needs to return:

            - index (int): the index of the kwargs in the kwargs_list.
            - success (bool): whether the function call was successful, whatever that
                definiton is.
            - epc_file (Path/str): the epc file path where the objects are stored.
            - uuid_list (List[str]): list of UUIDs of relevant objects.

        kwargs_list (List[Dict[Any]]): A list of keyword argument dictionaries that are
            used when calling the function.
        recombined_epc (Optional[Union[Path/str]]): A pathlib Path or path string of
            where the combined epc will be saved.
        consolidate (bool): if True and an equivalent part already exists in
            a model, it is not duplicated and the uuids are noted as equivalent.
        processes (Optional[int]): the number of processes in the pool to be created.

    Returns:
        success_list (List[bool]): A boolean list of successful function calls.
    """
    log.info("Multiprocessing function called with %s function.", function.__name__)

    # Creating temporary directories.
    tmp_dirs = []
    for i, kwargs in enumerate(kwargs_list):
        dirpath = tempfile.mkdtemp()
        kwargs["tmp_dir"] = dirpath
        kwargs["index"] = i
        tmp_dirs.append(Path(dirpath))
    log.info("Temporary directories created.")

    with Pool(processes = processes) as pool:
        log.info("Number of processes: %s", pool._processes)

        function_calls = [pool.apply_async(function, kwds = kwargs) for kwargs in kwargs_list]
        results = [func.get() for func in function_calls]

    log.info("Function calls complete.")

    # Sorting the results by the original kwargs_list index.
    results = list(sorted(results, key = lambda x: x[0]))

    success_list = [result[1] for result in results]
    epc_list = [result[2] for result in results]
    uuids_list = [result[3] for result in results]
    log.info("Number of successes: %s/%s.", sum(success_list), len(results))

    epc_file = Path(str(recombined_epc))
    if epc_file.is_file():
        model_recombined = Model(epc_file = str(epc_file))
    else:
        model_recombined = new_model(epc_file = str(epc_file))

    log.info("Creating the recombined epc file.")
    for i, epc in enumerate(epc_list):
        if epc is None:
            continue
        model = Model(epc_file = epc)
        uuids = uuids_list[i]
        if uuids is None:
            uuids = model.uuids()
        for uuid in uuids:
            model_recombined.copy_uuid_from_other_model(model, uuid = uuid, consolidate = consolidate)

    # Deleting temporary directories.
    for tmp_dir in tmp_dirs:
        rm_tree(tmp_dir)

    model_recombined.store_epc()

    log.info("Recombined epc file complete.")

    return success_list
