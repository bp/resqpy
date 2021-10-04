Getting started
===============

.. code-block:: python

    >>> import resqpy

A first step is typically to instantiate a :class:`resqpy.model.Model` object from your `.epc` file:

.. code-block:: python

    >>> from resqpy.model import Model
    >>> model = Model(epc_file='my_file.epc')
    <resqpy.model.Model at 0x7fdcd14e4700>

Models can be conveniently opened with the :class:`resqpy.model.ModelContext` context manager, to ensure file handles are closed properly upon exit:

.. code-block:: python

    >>> from resqpy.model import ModelContext
    >>> with ModelContext("my_model.epc") as model:
    >>>     print(model.uuids())

If you don't have any RESQML datasets, you can use the tiny datasets included in the example_data directory of the resqpy repository.

To list all the parts (high level objects) in the model:

.. code-block:: python

   for part in model.parts():
      
      part_type = model.type_of_part(part)
      title = model.citation_title_for_part(part)
      uuid = str(model.uuid_for_part(part))

      print(f'{title:<30s} {part_type:<35s} {uuid}')
