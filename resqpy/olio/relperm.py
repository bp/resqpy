"""relperm.py: class for dataframes of relative permeability data as RESQML objects.

Note that this module uses the obj_Grid2dRepresentation class in a way that was not envisaged
when the RESQML standard was defined; software that does not use resqpy is unlikely to be
able to do much with data stored in this way.
"""

# Nexus is a registered trademark of Halliburton

import logging

log = logging.getLogger(__name__)

import os
import numpy as np
import pandas as pd

from resqpy.olio.dataframe import DataFrame
import resqpy.olio.xml_et as rqet


class RelPerm(DataFrame):
    """Class for storing and retrieving a pandas dataframe of relative permeability data.

    note:
       inherits from DataFrame class
    """

    def __init__(self,
                 model,
                 uuid = None,
                 df = None,
                 uom_list = None,
                 realization = None,
                 phase_combo = None,
                 low_sal = False,
                 table_index = None,
                 title = 'relperm_table',
                 column_lookup_uuid = None,
                 uom_lookup_uuid = None):
        """Create a new RelPerm object from either a previously stored object or a pandas dataframe.

        arguments:
           phase_combo (str, optional): the combination of phases whose relative permeability behaviour is described.
              Options include 'water-oil', 'gas-oil' and 'gas-water'
           low_sal (boolean, optional): if True, indicates that the water-oil table contains the low-salinity data for
              relative permeability and capillary pressure
           table_index (int, optional): the index of the relative permeability
           table when multiple relative permeability tables are present. Note, indices should start at 1.

        note:
           see DataFrame class docstring for details of other arguments
        """

        # check that either a uuid OR dataframe has been provided
        if df is None and uuid is None:
            raise ValueError('either a uuid or a dataframe must be provided')

        # check extra_metadata for arguments if uuid is provided
        if uuid is not None:
            model_root = model.root(uuid = uuid)
            uuid_metadata_dict = rqet.load_metadata_from_xml(model_root)
            self.phase_combo = uuid_metadata_dict['phase_combo']
            self.low_sal = uuid_metadata_dict['low_sal']
            self.table_index = uuid_metadata_dict['table_index']

        else:
            # check that 'phase_combo' parameter is valid
            self.phase_combo = phase_combo
            self.low_sal = low_sal
            self.table_index = table_index

            df.columns = [x.capitalize() for x in df.columns]
            if 'Pc' in df.columns and df.columns[-1] != 'Pc':
                raise ValueError('Pc', 'capillary pressure data should be in the last column of the dataframe')

            processed_phase_combo_checks = {
                ('oil', 'water'): self.__water_oil_error_check,
                ('gas', 'oil'): self.__gas_oil_error_check,
                ('gas', 'water'): self.__gas_water_error_check,
                ('None',): self.__no_phase_combo_error_check
            }

            processed_phase_combo = tuple(sorted(set([x.strip() for x in str(self.phase_combo).split('-')])))
            if processed_phase_combo not in processed_phase_combo_checks.keys():
                raise ValueError('invalid phase_combo provided')
            # check that table_index is >= 1
            if table_index is not None and table_index < 1:
                raise ValueError('table_index cannot be less than 1')
            # check that the column names and order are as expected
            processed_phase_combo_checks.get(processed_phase_combo)(df)
            # ensure that missing capillary pressure values are stored as np.nan
            if 'Pc' in df.columns:
                df['Pc'].replace('None', np.nan, inplace = True)
            # convert all values in the dataframe to numeric type
            df[df.columns] = df[df.columns].apply(pd.to_numeric, errors = 'coerce')
            # ensure that no other column besides Pc has missing values
            cols_no_pc = [x for x in df.columns if 'Pc' != x]
            for col in cols_no_pc:
                if (df[col].isnull().sum() > 0) or ('None' in list(df[col])):
                    raise Exception(f'missing values found in {col} column')

            # check that Sw, Kr and Pc values are monotonic and that the Sw and Kr values are within the range 0-1
            sat_col = [x for x in df.columns if x[0] == 'S'][0]
            relp_corr_col = [x for x in df.columns if x == 'Kr' + sat_col[-1]][0]
            relp_opp_col = [x for x in df.columns if (x[0:2] == 'Kr') & (x[-1] != sat_col[-1])][0]
            if not (df[sat_col].is_monotonic and df[relp_corr_col].is_monotonic and df[
                relp_opp_col].is_monotonic_decreasing) and not \
                    (df[sat_col].is_monotonic_decreasing and df[relp_corr_col].is_monotonic_decreasing and
                     df[relp_opp_col].is_monotonic):
                raise ValueError(f'{sat_col, relp_corr_col, relp_opp_col} combo is not monotonic')
            if 'Pc' in df.columns:
                if not df['Pc'].dropna().is_monotonic and not df['Pc'].dropna().is_monotonic_decreasing:
                    raise ValueError('Pc values are not monotonic')
            for col in ['Sw', 'Sg', 'So', 'Krw', 'Krg', 'Kro']:
                if col in df.columns:
                    if df[col].min() < 0 or df[col].max() > 1:
                        raise ValueError(f'{col} is not within the range 0-1')

        super().__init__(model,
                         uuid = uuid,
                         df = df,
                         uom_list = uom_list,
                         realization = realization,
                         title = title,
                         column_lookup_uuid = column_lookup_uuid,
                         uom_lookup_uuid = uom_lookup_uuid,
                         extra_metadata = {
                             'relperm_table': 'true',
                             'phase_combo': self.phase_combo,
                             'low_sal': str(self.low_sal).lower(),
                             'table_index': str(self.table_index)
                         })

    @staticmethod
    def __error_check(expected_cols, sat_cols, df):
        if df.columns[0] not in sat_cols or len(set(df.columns).intersection(sat_cols)) != 1:
            raise ValueError('incorrect saturation column name and/or multiple saturation columns exist')
        if not set(df.columns).issubset(expected_cols):
            raise ValueError(f'incorrect column name(s) {set(df.columns).difference(expected_cols)}')

    def __water_oil_error_check(self, df):
        expected_cols = {'Sw', 'So', 'Krw', 'Kro', 'Pc'}
        sat_cols = {'Sw', 'So'}
        self.__error_check(expected_cols, sat_cols, df)

    def __gas_oil_error_check(self, df):
        expected_cols = {'Sg', 'So', 'Krg', 'Kro', 'Pc'}
        sat_cols = {'Sg', 'So'}
        self.__error_check(expected_cols, sat_cols, df)

    def __gas_water_error_check(self, df):
        expected_cols = {'Sg', 'Sw', 'Krg', 'Krw', 'Pc'}
        sat_cols = {'Sg', 'Sw'}
        self.__error_check(expected_cols, sat_cols, df)

    def __no_phase_combo_error_check(self, df):
        if df.columns[0] not in ['Sw', 'Sg', 'So'] or len(set(df.columns).intersection({'Sw', 'Sg', 'So'})) != 1:
            raise ValueError('incorrect saturation column name and/or multiple saturation columns exist')
        if set(df.columns).issubset({'Sw', 'So', 'Krw', 'Kro', 'Pc'}) and len(set(df.columns)) >= 3:
            self.phase_combo = 'water-oil'
        elif set(df.columns).issubset({'Sg', 'So', 'Krg', 'Kro', 'Pc'}) and len(set(df.columns)) >= 3:
            self.phase_combo = 'gas-oil'
        elif set(df.columns).issubset({'Sg', 'Sw', 'Krg', 'Krw', 'Pc'}) and len(set(df.columns)) >= 3:
            self.phase_combo = 'gas-water'
        else:
            raise Exception('unexpected number of columns and/or column headers')

    def interpolate_point(self, saturation, kr_or_pc_col):
        """Returns a tuple of the saturation value and the corresponding interpolated rel. perm. or cap. pressure value.

        arguments:
           saturation (float): the saturation at which the relative permeability or cap. pressure will be interpolated
           kr_or_pc_col (str): the column name of the parameter to be interpolated

        returns:
           tuple of float, the first element is the saturation and the second element is the interpolated value

        note:
           A simple linear interpolation is performed.
        """
        df = self.df.copy()
        if kr_or_pc_col.capitalize() not in df.columns or kr_or_pc_col.capitalize() == df.columns[0]:
            raise ValueError('incorrect column name provided for interpolation')
        if kr_or_pc_col == 'PC':
            df = df[df['PC'].notnull()]
        sat_col = df.columns[0]
        # ensure that the saturation values are monotonically increasing
        df = df.sort_values(by = sat_col)
        x = df[sat_col]
        y = df[kr_or_pc_col]
        x_new = saturation
        if x_new < x.min() or x_new > x.max():
            raise ValueError('saturation value is outside the interpolation range')
        y_new = np.interp(x_new, x, y)
        return saturation, y_new

    def df_to_text(self, filepath, filename):
        """Creates a text file from a dataframe of relative permeability and capillary pressure data.

        arguments:
           filepath (str): location where new text file is written to
           filename (str): name of the new text file

        returns:
           tuple of float, the first element is the saturation and the second element is the interpolated value

        note:
           Only Nexus compatible text files are currently supported. Text files that are compatible with other reservoir
              simulators may be supported in the future.
        """
        df = self.df.copy()
        ascii_file = os.path.join(filepath, filename + '.dat')
        df.columns = map(str.upper, df.columns)
        if {'KRW', 'KRO'}.issubset(set(df.columns)):
            if self.low_sal:
                table_name_keyword = 'WOTABLE (LOW_SAL)\n'
            else:
                table_name_keyword = 'WOTABLE\n'
            df.rename(columns = {'SW': 'SW', 'KRW': 'KRW', 'KRO': 'KROW', 'PC': 'PCWO'}, inplace = True)
        elif {'KRG', 'KRO'}.issubset(set(df.columns)):
            table_name_keyword = 'GOTABLE\n'
            df.rename(columns = {'SG': 'SG', 'KRG': 'KRG', 'KRO': 'KROG', 'PC': 'PCGO'}, inplace = True)
        elif {'KRW', 'KRW'}.issubset(set(df.columns)):
            table_name_keyword = 'GWTABLE\n'
            df.rename(columns = {'SG': 'SG', 'KRG': 'KRG', 'KRW': 'KRWG', 'PC': 'PCGW'}, inplace = True)
        else:
            raise Exception('incorrect rel. perm. column combination encountered')

        if filename + '.dat' not in os.listdir(filepath):
            with open(ascii_file, 'w') as f:
                f.write(table_name_keyword)
                df_str = df.to_string(na_rep = '', index = False)
                f.write(df_str)
                f.close()
                print(f'Created new DAT file: {filename} at {filepath}')
        else:
            with open(ascii_file, 'a') as f:
                f.write('\n\n')
                f.write(table_name_keyword)
                df_str = df.to_string(na_rep = '', index = False)
                f.write(df_str)
                f.close()
                print(f'Appended to DAT file: {filename} at {filepath}')

    def write_hdf5_and_create_xml(self):
        """Write relative permeability table data to hdf5 file and create xml for dataframe objects."""
        super().write_hdf5_and_create_xml()
        mesh_root = self.mesh.root
        # create an xml of extra metadata to indicate that this is a relative permeability table
        rqet.create_metadata_xml(mesh_root, self.extra_metadata)


def text_to_relperm_dict(relperm_data, is_file = True):
    """Return dict of dataframes with relative permeability and capillary pressure data and phase combinations.

    arguments:
       relperm_data (str): relative or full path of the text file to be processed or string of relative permeability data
       is_file (boolean): if True, indicates that a text file of relative permeability data has been provided. Default value is True

    returns:
       dict, each element in the dictionary contains a dataframe, with saturation and rel. permeability/capillary pressure
       data, and the phase combination being described

    note:
       Only Nexus compatible text files are currently supported. Text files from other reservoir simulators may be
          supported in the future.
    """
    if is_file:
        with open(relperm_data) as f:
            string_original = f.read()
    else:
        string_original = relperm_data
    # split the string based on newlines and remove any comments or other escape characters
    escapes = ''.join([chr(char) for char in range(1, 32)])
    string_formatted = [x.strip(escapes).split(' ') for x in filter(None, string_original.split('\n')) if '!' not in x]
    # remove all empty strings
    data = [list(filter(None, x)) for x in string_formatted]
    # get indices of start of each new relperm table based on Nexus keywords
    table_start_positions = [
        i for i, l in enumerate(data) if len({'WOTABLE', 'GOTABLE', 'GWTABLE'}.intersection(set(l))) == 1
    ]
    df_cols_dict = {
        'SW': 'Sw',
        'SG': 'Sg',
        'KRW': 'Krw',
        'KRG': 'Krg',
        'KROW': 'Kro',
        'KROG': 'Kro',
        'KRWG': 'Krw',
        'PCWO': 'Pc',
        'PCGO': 'Pc',
        'PCGW': 'Pc',
        'PC': 'Pc'
    }
    relperm_table_idx = 1
    rel_perm_dict = {}
    for i, l in enumerate(table_start_positions):
        key = 'relperm_table' + str(relperm_table_idx)
        rel_perm_dict[key] = {}
        relperm_table_idx += 1
        if 'WOTABLE' in data[l]:
            phase_combo = 'water-oil'
            rel_perm_dict[key]['phase_combo'] = phase_combo
        elif 'GOTABLE' in data[l]:
            phase_combo = 'gas-oil'
            rel_perm_dict[key]['phase_combo'] = phase_combo
        elif 'GWTABLE' in data[l]:
            phase_combo = 'gas-water'
            rel_perm_dict[key]['phase_combo'] = phase_combo
        else:
            raise Exception('incorrect table key word encountered')
        if i < (len(table_start_positions) - 1):
            table_end = table_start_positions[i + 1]
        else:
            table_end = len(data)
        table_cols = data[l + 1]
        table_rows = data[l + 2:table_end]
        df = pd.DataFrame(table_rows, columns = table_cols)
        df.columns = df.columns.map(df_cols_dict)
        sat_col = [x for x in df.columns if 'S' in x][0]
        df = df.apply(pd.to_numeric, errors = 'coerce')
        df = df[df[sat_col].notnull()]
        rel_perm_dict[key]['df'] = df
    return rel_perm_dict


def relperm_parts_in_model(model,
                           phase_combo = None,
                           low_sal = None,
                           table_index = None,
                           title = None,
                           related_uuid = None):
    """Returns list of part names within model that are representing RelPerm dataframe support objects.

    arguments:
       model (model.Model): the model to be inspected for dataframes
       phase_combo (str, optional): the combination of phases whose relative permeability behaviour is described.
          Options include 'water-oil', 'gas-oil', 'gas-water', 'oil-water', 'oil-gas' and 'water-gas'
       low_sal (boolean, optional): if True, indicates that the water-oil table contains the low-salinity data for
          relative permeability and capillary pressure
       table_index (int, optional): the index of the relative permeability table when multiple relative permeability
          tables are present. Note, indices should start at 1.
       title (str, optional): if present, only parts with a citation title exactly matching will be included
       related_uuid (uuid, optional): if present, only parts relating to this uuid are included

    returns:
       list of str, each element in the list is a part name, within model, which is representing the support for a RelPerm object
    """
    extra_metadata_orig = {
        'relperm_table': 'true',
        'phase_combo': phase_combo,
        'low_sal': low_sal,
        'table_index': table_index
    }
    extra_metadata = {k: str(v).lower() for k, v in extra_metadata_orig.items() if v is not None}
    df_parts_list = model.parts(obj_type = 'Grid2dRepresentation',
                                title = title,
                                extra = extra_metadata,
                                related_uuid = related_uuid)

    return df_parts_list
