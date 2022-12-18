"""read_nexus_fault.py: functions for reading Nexus fault definition data from an ascii file."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import os
import re
import unicodedata
import numpy as np
import pandas as pd


def load_nexus_fault_mult_table(file_name):
    """Reads a Nexus (!) format file containing one or more MULT keywords and returns a dataframe with the MULT rows."""

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    dfs = []

    num_tables = 0
    ISTABLE = False
    ISRECORD = False
    name = face = None
    grid = 'ROOT'  ## nexus default grid

    face_dict = {'TX': 'I', 'TY': 'J', 'TZ': 'K', 'TI': 'I', 'TJ': 'J', 'TK': 'K'}

    if os.path.isfile(file_name):
        chunks = []
        with open(file_name) as f:
            for line in f:
                if len(line.strip()):
                    if (not line.strip()[0] == '!') & (not line.strip()[0] == 'C'):
                        line = line.partition('!')[0]  # removing trailing comments
                        # line = line.partition('C')[0]  # removing trailing comments
                        tokens = line.split()
                        if ISTABLE:
                            if is_number(tokens[0]):
                                ISRECORD = True

                            if ISRECORD and (not is_number(tokens[0])):
                                data = chunks[0:]
                                d_elems = np.array([np.array(data[i].split()) for i in range(len(data))])
                                # fill empty elements with zero
                                lens = np.array([len(i) for i in d_elems])
                                # Mask of valid places in each row
                                mask = np.arange(lens.max()) < lens[:, None]
                                # Setup output array and put elements from data into masked positions
                                outdata = np.zeros(mask.shape, dtype = d_elems.dtype)
                                outdata[mask] = np.concatenate(d_elems)
                                df = pd.DataFrame(outdata)
                                for column in df.columns:
                                    df[column] = pd.to_numeric(df[column], errors = 'ignore')
                                df.columns = ['i1', 'i2', 'j1', 'j2', 'k1', 'k2', 'mult']
                                df['grid'] = grid
                                df['name'] = name
                                df['face'] = face
                                dfs.append(df)
                                num_tables += 1

                                ISTABLE = False
                                ISRECORD = False
                                chunks = []

                        if ISTABLE:
                            if re.match("(.*)GRID(.*)", tokens[0]):
                                if len(tokens) > 0:
                                    grid = tokens[1]
                            elif re.match("(.*)FNAME(.*)", tokens[0]):
                                if len(tokens) > 0:
                                    name = tokens[1]
                            else:
                                if re.match(r"^MULT$", tokens[0]):
                                    ISTABLE = False
                                    ISRECORD = False
                                    chunks = []
                                else:
                                    chunks.append(line.strip())

                        if re.match(r"^MULT$", tokens[0]):
                            if len(tokens) > 0:
                                face = face_dict[tokens[1]]
                            if 'MINUS' in tokens:
                                face += '-'  # indicates data apply to 'negative' faces of specified cells
                            grid = 'ROOT'  # nexus default
                            name = 'NONE'
                            ISTABLE = True

            else:
                if ISTABLE:
                    if ISRECORD:
                        data = chunks[0:]
                        d_elems = np.array([np.array(data[i].split()) for i in range(len(data))])
                        # fill empty elements with zero
                        lens = np.array([len(i) for i in d_elems])
                        # Mask of valid places in each row
                        mask = np.arange(lens.max()) < lens[:, None]
                        # Setup output array and put elements from data into masked positions
                        outdata = np.zeros(mask.shape, dtype = d_elems.dtype)
                        outdata[mask] = np.concatenate(d_elems)
                        df = pd.DataFrame(outdata)
                        for column in df.columns:
                            df[column] = pd.to_numeric(df[column], errors = 'ignore')
                        df.columns = ['i1', 'i2', 'j1', 'j2', 'k1', 'k2', 'mult']
                        df['grid'] = grid
                        df['name'] = name
                        df['face'] = face
                        dfs.append(df)
                        num_tables += 1

                        ISTABLE = False
                        ISRECORD = False
                        chunks = []

    fault_df = pd.concat(dfs).reset_index(drop = True)

    convert_dict = {'i1': int, 'i2': int, 'j1': int, 'j2': int, 'k1': int, 'k2': int, 'mult': float}
    fault_df = fault_df.astype(convert_dict)

    return fault_df


def load_nexus_fault_mult_table_new(file_name):
    """Reads a Nexus (!) format file containing one or more MULT keywords and returns a dataframe with the MULT rows."""

    return load_nexus_fault_mult_table(file_name)
