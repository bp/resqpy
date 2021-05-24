Getting started
===============

.. code-block:: python

    >>> import resqpy

A first step is typically to instantiate a :class:`resqpy.model.Model` object from your `.epc` file:

.. code-block:: python

    >>> from resqpy.model import Model
    >>> model = Model(epc_file='my_file.epc')
    <resqpy.model.Model at 0x7fdcd14e4700>

To iterate over all wells in the model:

.. code-block:: python

    for well in model.wells():
       print(well.title)

       for trajectory in well.trajectories():
          print(trajectory.title)

          for frame in trajectory.wellbore_frames():
             print(frame.title)

             # Measured depths
             mds = frame.node_mds

             # Logs
             log_collection = frame.logs
             for log in log_collection.logs():
                values = log.values()
