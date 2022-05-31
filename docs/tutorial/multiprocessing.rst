Multiprocessing
===============

This tutorial is about using multiprocessing with specific resqpy functions to speed up multiple
function calls.

You should edit the file paths in the examples to point to your own files.

Installing Dask
---------------
To use the multiprocesing module, Dask needs to be installed in the Python environment because it is
not a dependency of the project. Dask is a flexible open-source Python library for parallel
computing. It scales Python code from multi-core local machines to large distributed clusters
on-prem or in the cloud.

Dask contains multiple modules but only the distributed module is needed here. Dask Distributed can
be installed using pip, conda, or from source.

Pip
~~~

.. code-block::

    python -m pip install dask distributed

Conda
~~~~~

.. code-block::

    conda install dask distributed -c conda-forge

Source
~~~~~~

.. code-block::

    git clone https://github.com/dask/distributed.git
    cd distributed
    python -m pip install .

If using a Job Queue Cluster, Dask Jobqueue must also be installed. This can be installed in the
same ways.

Pip
~~~

.. code-block::

    python -m pip install dask-jobqueue

Conda
~~~~~

.. code-block::

    conda install dask-jobqueue -c conda-forge

Source
~~~~~~

.. code-block::

    git clone https://github.com/dask/dask-jobqueue.git
    cd dask-jobqueue
    python -m pip install .


Cluster & Client Setup
----------------------
If using a local machine, a `LocalCluster` must be setup. If using a job queing system, a
`JobQueueCluster` can be used such as an `SGECluster`, `SLURMCluster`, `PBSCluster`, `LSFCluster`
etc. Full details can be found at https://docs.dask.org/en/latest/deploying.html

A client can also be setup to provide a live feedback dashboard or to capture diagnosics, which is
explained in the next section.

Local Cluster
~~~~~~~~~~~~~
Documentation of creating a `LocalCluster` can be found at
https://distributed.dask.org/en/stable/api.html#distributed.LocalCluster

.. code-block:: python

    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster()
    client = Client(cluster)

Job Queue Cluster Example
~~~~~~~~~~~~~~~~~~~~~~~~~
As an example, an SGE Cluster can be setup using Dask Jobqueue. Documentation of creating a
`JobQueueCluster` can be found at https://jobqueue.dask.org/en/latest/api.html

.. code-block:: python

    from dask.distributed import client
    from dask_jobqueue import SGECluster

    cluster = SGECluster(
        processes=1,        # Number of workers per job.
        cores=96,           # Total amount of physical cores for all workers.
        memory="360 GiB",   # Usable memory per node.
        scheduler_options={"dashboard_address": ":0"}   # Other scheduler options.
    )
    client = Client(cluster)


Viewing the Client
------------------
If using a Local Cluster, the client dashboard is typically served at http://localhost:8787/status ,
but may be served elsewhere if this port is taken. The address of the dashboard will be displayed if
you are in a Jupyter Notebook, or can be queried from client.dashboard_link.

Some clusters restrict the ports that are visible to the outside world. These ports may include the
default port for the web interface, 8787. There are a few ways to handle this:

* Open port 8787 to the outside world. Often this involves asking your cluster administrator.
* Use a different port that is publicly accessible using the `scheduler_options` argument, like above.
* Use fancier techniques, like Port Forwarding

You can capture some of the same information that the dashboard presents for offline processing
using the `Client.get_task_stream` and `Client.profile` methods. These capture the start and stop
time of every task and transfer, as well as the results of a statistical profiler. More info on this
can be found at https://docs.dask.org/en/stable/diagnostics-distributed.html#capture-diagnostics

Uplading Packages/ Files to the Workers
---------------------------------------
If using a Job Queue Cluster, the resqpy package may need to be uploaded for the workers to use. A
dependency file that contains the path of the installed resqpy package or the location of a local
git clone of the repo can be uploaded to the client.

.. code-block:: python

    dependencies = """
    import sys
    sys.path.insert(0, "path/to/local/resqpy/clone")
    """

    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "dependencies.py")
        with open(filename, "w") as f:
            f.write(dependencies)

        client.wait_for_workers()
        client.upload_file(filename)

Environment variables may also need to be set such as the Numba thread limit, which can be done by
running a defined function.

.. code-block:: python

    def set_numba_threads():
        os.environ["NUMBA_NUM_THREADS"] = "1"

    client.run(set_numba_threads)


Resqpy Wrapper Functions
------------------------
To run the multiprocessing function, a wrapper function for the corresponding resqpy function is
required. These can be found within the `multiprocessing.wrappers` module. Currently there is only a
wrapper function for the `find_faces_to_represent_surface_regular` function, however any wrapper
function can be created, providing that it returns the following:

* index (int): the index passed to the function.
* success (bool): whether the function call was successful, whatever that definiton is.
* epc_file (str): the epc file path where the objects are stored.
* uuid_list (List[str]): list of UUIDs of relevant objects.

The multiprocessing function will combine all of the objects that have their UUIDs returned, into a
single epc file.

Calling the Multiprocessing Function
------------------------------------
The multiprocessing function must receive the following arguments:

* function (Callable): the wrapper function to be called, that must return the items described
  above.
* kwargs_list (List[Dict[Any]]): A list of keyword argument dictionaries that are used when calling
  the function.
* recombined_epc (Path/str): A pathlib Path or path string of where the combined epc will be saved.
* cluster (LocalCluster/JobQueueCluster): the relevant cluster, as explained above.
* consolidate (bool): if True and an equivalent part already exists in a model, it is not duplicated
  and the uuids are noted as equivalent.

.. code-block:: python

    from resqpy.multiprocessing import function_multiprocessing

    success_list = function_multiprocessing(func, kwargs_list, recombined_epc, cluster=cluster)

A list of successes from the wrapper function in order of their call is returned.
