"""_context.py: ModelContext class."""

import logging

log = logging.getLogger(__name__)

import os
from typing import Optional

import resqpy.model
from resqpy.model import Model, new_model


class ModelContext:
    """Context manager for easy opening and closing of resqpy models.

    When a model is opened this way, any open file handles are safely closed
    when the "with" clause exits. Optionally, the epc can be written back to
    disk upon exit.

    Example::

        with ModelContext("my_model.epc", mode="rw") as model:
            print(model.uuids())

    Note:
        The "write_hdf5" and "create_xml" methods of individual resqpy objects
        still need to be invoked as usual.
    """

    def __init__(self, epc_file, mode = "r") -> None:
        """Open a resqml file, safely closing file handles upon exit.

        arguments:
            epc_file (str): path to existing resqml file
            mode (str, default 'r'): one of "read", "read/write", "create", or shorthands "r", "rw", "c"

        notes:
            the modes operate as follows:
            - In "read" mode, an existing epc file is opened; any changes are not
            saved to disk automatically, but can still be saved by calling
            `model.store_epc()`;
            - In "read/write" mode, changes are written to disk when the context exists;
            - In "create" mode, a new model is created and saved upon exit; any pre-existing
            model will be deleted
        """

        # Validate mode
        modes_mapping = {"r": "read", "rw": "read/write", "c": "create"}
        mode = modes_mapping.get(mode, mode)
        if mode not in modes_mapping.values():
            raise ValueError(f"Unexpected mode '{mode}'")

        self.epc_file = epc_file
        self.mode = mode
        self._model: Optional[Model] = None

    def __enter__(self) -> Model:
        """Enter the runtime context, return a model."""

        if self.mode in ["read", "read/write"]:
            if not os.path.exists(self.epc_file):
                raise FileNotFoundError(self.epc_file)
            self._model = Model(epc_file = str(self.epc_file))

        else:
            assert self.mode == "create"
            for file in [self.epc_file, self.epc_file[:-4] + '.h5']:
                if os.path.exists(file):
                    os.remove(file)
                    log.info('old file deleted: ' + str(file))
            self._model = new_model(self.epc_file)

        return self._model

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Exit the runtime context, close the model."""

        # Only write to disk if no exception has occured
        if self.mode in ["read/write", "create"] and exc_type is None:
            self._model.store_epc()

        # Release file handles
        self._model.h5_release()
