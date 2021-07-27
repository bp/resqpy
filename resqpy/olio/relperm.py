<<<<<<< HEAD
"""relperm.py: class for storing and retrieving dataframes of relative permeability
data as RESQML objects.

   note that this module uses the obj_Grid2dRepresentation class in a way that was not envisaged
   when the RESQML standard was defined; software that does not use resqpy is unlikely to be
   able to do much with data stored in this way
"""
import numpy as np
import pandas as pd
from scipy import interpolate
import os
import logging
import resqpy.olio.xml_et as rqet
from resqpy.olio.dataframe import DataFrame

version = '27th July 2021'

log = logging.getLogger(__name__)
log.debug(f'dataframe.py version {version}')


class RelPerm(DataFrame):
    """Class for storing and retrieving a pandas dataframe of relative permeability data.

    note:
       inherits from DataFrame class
    """

    def __init__(
            self,
            model,
            uuid=None,
            df=None,
            uom_list=None,
            realization=None,
            phase_combo=None,
            low_sal=False,
            table_index=None,
            title='relperm_table',
            column_lookup_uuid=None,
            uom_lookup_uuid=None):
        """Create a new RelPerm object from either a previously stored property or a pandas dataframe.

        arguments:
           phase_combo (str, optional): the combination of phases whose relative
           permeability behaviour is described. Options include 'water-oil', 'gas-oil' and
           'gas-water'
           low_sal (boolean, optional): if True, indicates that the water-oil table contains
           the low-salinity data for relative permeability and capillary pressure
           table_index (int, optional): the index of the relative permeability
           table when multiple relative permeability tables are present. Note, indices should start at 1.

        note:
           see DataFrame class docstring for details of other arguments
        """

        # check that either a uuid OR dataframe has been provided
        assert uuid is not None or df is not None, 'either a uuid or a dataframe must be provided'

        # check that 'phase_combo' parameter is valid
        processed_phase_combo = set([x.strip() for x in str(phase_combo).split('-')])
        assert processed_phase_combo in [{'water', 'oil'}, {'gas', 'oil'}, {'gas', 'water'},
                                         {'None'}], 'invalid phase_combo provided'

        # check that table_index is >= 1
        if table_index is not None:
            assert table_index >= 1, 'table_index cannot be less than 1'

        # check that the column names and order are as expected
        if df is not None:
            df.columns = [x.capitalize() for x in df.columns]
            if 'Pc' in df.columns:
                assert df.columns[-1] == 'Pc', 'capillary pressure data should be in the last column of the dataframe'
            if phase_combo is not None:
                if processed_phase_combo == {'water', 'oil'}:
                    expected_cols = {'Sw', 'So', 'Krw', 'Kro', 'Pc'}
                    sat_cols = {'Sw', 'So'}
                    assert df.columns[0] in sat_cols and len(set(df.columns).intersection(
                        sat_cols)) == 1, 'incorrect saturation column name and/or multiple saturation columns exist'
                    assert set(df.columns).issubset(
                        expected_cols), f'incorrect column name(s) {set(df.columns).difference(expected_cols)} \
                        in water-oil rel. perm table'
                elif processed_phase_combo == {'gas', 'oil'}:
                    expected_cols = {'Sg', 'So', 'Krg', 'Kro', 'Pc'}
                    sat_cols = {'Sg', 'So'}
                    assert df.columns[0] in sat_cols and len(set(df.columns).intersection(
                        sat_cols)) == 1, 'incorrect saturation column name and/or multiple saturation columns exist'
                    assert set(df.columns).issubset(
                        expected_cols), f'incorrect column name(s) {set(df.columns).difference(expected_cols)} \
                         in gas-oil rel. perm table'
                elif processed_phase_combo == {'gas', 'water'}:
                    expected_cols = {'Sg', 'Sw', 'Krg', 'Krw', 'Pc'}
                    sat_cols = {'Sg', 'Sw'}
                    assert df.columns[0] in sat_cols and len(set(df.columns).intersection(
                        sat_cols)) == 1, 'incorrect saturation column name and/or multiple saturation columns exist'
                    assert set(df.columns).issubset(
                        expected_cols), f'incorrect column name(s) {set(df.columns).difference(expected_cols)} \
                         in gas-oil rel. perm table'
            elif phase_combo is None:
                assert df.columns[0] in ['Sw', 'Sg', 'So'] and len(set(df.columns).intersection({'Sw', 'Sg',
                                                                                                 'So'})) == 1, \
                    'incorrect saturation column name and/or multiple saturation columns exist'
                if set(df.columns).issubset({'Sw', 'So', 'Krw', 'Kro', 'Pc'}) and len(set(df.columns)) >= 3:
                    phase_combo = 'water-oil'
                elif set(df.columns).issubset({'Sg', 'So', 'Krg', 'Kro', 'Pc'}) and len(set(df.columns)) >= 3:
                    phase_combo = 'gas-oil'
                elif set(df.columns).issubset({'Sg', 'Sw', 'Krg', 'Krw', 'Pc'}) and len(set(df.columns)) >= 3:
                    phase_combo = 'gas-water'
                else:
                    raise Exception('unexpected number of columns and/or column headers')

            # ensure that missing capillary pressure values are stored as np.nan
            for col in df.columns:
                if col.capitalize() == 'Pc':
                    df[col].replace('None', np.nan, inplace=True)

            # convert all values in the dataframe to numeric type
            df_cols = df.columns
            df[df_cols] = df[df_cols].apply(pd.to_numeric, errors='coerce')

            # ensure that no other column besides Pc has missing values
            for col in df.columns:
                if col.capitalize != 'Pc':
                    continue
                elif (df[col].isnull().sum() > 0) | ('None' in list(df[col])):
                    raise Exception(f'missing values found in {col} column')

            # check that Sw, Kr and Pc values are monotonic and that the Sw and Kr values are within the range 0-1
            sat_col = [x for x in df.columns if x[0] == 'S'][0]
            relp_corr_col = [x for x in df.columns if x == 'Kr' + sat_col[-1]][0]
            relp_opp_col = [x for x in df.columns if (x[0:2] == 'Kr') & (x[-1] != sat_col[-1])][0]
            assert (df[sat_col].is_monotonic and df[relp_corr_col].is_monotonic and df[
                relp_opp_col].is_monotonic_decreasing) or \
                (df[sat_col].is_monotonic_decreasing and df[relp_corr_col].is_monotonic_decreasing and
                    df[relp_opp_col].is_monotonic), f'{sat_col, relp_corr_col, relp_opp_col} combo is not monotonic'
            if 'Pc' in df.columns:
                assert df['Pc'].dropna().is_monotonic or df[
                    'Pc'].dropna().is_monotonic_decreasing, 'Pc values are not monotonic'
            for col in ['Sw', 'Sg', 'So', 'Krw', 'Krg', 'Kro']:
                if col in df.columns:
                    assert df[col].min() >= 0 and df[col].max() <= 1, f'{col} is not within the range 0-1'

        super().__init__(model,
                         uuid=uuid,
                         support_root=None,
                         df=df,
                         uom_list=uom_list,
                         realization=realization,
                         title=title,
                         column_lookup_uuid=column_lookup_uuid,
                         uom_lookup_uuid=uom_lookup_uuid)
        self.phase_combo = phase_combo
        self.low_sal = low_sal
        self.table_index = table_index

    def interpolate_point(self, saturation, kr_or_pc_col):
        """Returns a tuple of the saturation value and the corresponding
        interpolated relative permeability or capillary pressure value.

        arguments:
            saturation (float): the saturation at which the relative permeability or cap. pressure will be interpolated
            kr_or_pc_col (str): the column name of the parameter to be interpolated
        returns:
            tuple of float, the first element is the saturation and the second element is the interpolated value
           """
        df = self.df.copy()
        assert kr_or_pc_col.capitalize() in df.columns and kr_or_pc_col.capitalize() != df.columns[
            0], 'incorrect column name provided for interpolation'
        if kr_or_pc_col == 'PC':
            df = df[df['PC'].notnull()]
        sat_col = df.columns[0]
        x = df[sat_col]
        y = df[kr_or_pc_col]
        f = interpolate.interp1d(x, y, kind='linear', assume_sorted=True)
        x_new = saturation
        assert x.min() <= x_new <= x.max(), 'saturation value is outside the interpolation range'
        y_new = f(x_new)
        return saturation, y_new.item()

    def df_to_text(self, filepath, filename):
        """Creates text file from a dataframe of relative permeability and capillary pressure data.

        arguments:
            filepath (str): location where new text file is written to
            filename (str): name of the new text file
        returns:
            tuple of float, the first element is the saturation and the second element is the interpolated value
           """
        df = self.df.copy()
        ascii_file = os.path.join(filepath, filename + '.dat')
        df.columns = map(str.upper, df.columns)
        if {'KRW', 'KRO'}.issubset(set(df.columns)):
            if self.low_sal:
                table_name_keyword = 'WOTABLE (LOW_SAL)\n'
            else:
                table_name_keyword = 'WOTABLE\n'
            df.rename(columns = {'SW': 'SW', 'KRW': 'KRW', 'KRO': 'KROW', 'PC': 'PCWO'}, inplace=True)
        elif {'KRG', 'KRO'}.issubset(set(df.columns)):
            table_name_keyword = 'GOTABLE\n'
            df.rename(columns = {'SG': 'SG', 'KRG': 'KRG', 'KRO': 'KROG', 'PC': 'PCGO'}, inplace=True)
        elif {'KRW', 'KRW'}.issubset(set(df.columns)):
            table_name_keyword = 'GWTABLE\n'
            df.rename(columns = {'SG': 'SG', 'KRG': 'KRG', 'KRW': 'KRWG', 'PC': 'PCGW'}, inplace=True)
        else:
            raise Exception('incorrect rel. perm. column combination encountered')

        if filename + '.dat' not in os.listdir(filepath):
            with open(ascii_file, 'w') as f:
                f.write(table_name_keyword)
                df_str = df.to_string(na_rep='', index=False)
                f.write(df_str)
                f.close()
                print(f'Created new DAT file: {filename} at {filepath}')
        else:
            with open(ascii_file, 'a') as f:
                f.write('\n\n')
                f.write(table_name_keyword)
                df_str = df.to_string(na_rep='', index=False)
                f.write(df_str)
                f.close()
                print(f'Appended to DAT file: {filename} at {filepath}')

    def write_hdf5_and_create_xml(self):
        """Write relative permeability table data to hdf5 file and create xml for RESQML
        objects to represent dataframe."""

        super().write_hdf5_and_create_xml()
        # note: time series xml must be created before calling this method
        mesh_root = self.mesh.root
        # create an xml of extra metadata to indicate that this is a relative permeability table
        rqet.create_metadata_xml(mesh_root, {'relperm_table': 'true'})


def text_to_relperm_dict(filepath):
    """
    Returns a dictionary that contains dataframes with relative permeability and capillary pressure data and
    phase combinations.

    arguments:
    filepath (str): relative or full path of the text file to be processed

    returns:
    dict, each element in the dictionary contains a dataframe, with saturation and rel. permeability/capillary pressure
    data, and the phase combination being described
    """
    with open(filepath) as f:
        # create list of rows of the original ascii file with blank lines removed
        data = list(filter(None, [list(filter(None, x.strip('\n').split(' '))) for x in f.readlines()]))
        # remove comments
        data = [x for x in data if ('!' not in x)]

    # get indices of start of each new relperm table based on Nexus keywords
    table_start_positions = [i for i, l in enumerate(data) if
                             len({'WOTABLE', 'GOTABLE', 'GWTABLE'}.intersection(set(l))) == 1]
    df_cols_dict = {'SW': 'Sw', 'SG': 'Sg', 'KRW': 'Krw', 'KRG': 'Krg', 'KROW': 'Kro',
                    'KROG': 'Kro', 'KRWG': 'Krw', 'PCWO': 'Pc', 'PCGO': 'Pc', 'PCGW': 'Pc'}
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
        table_rows = data[l + 2: table_end]
        df = pd.DataFrame(table_rows, columns=table_cols)
        df.columns = df.columns.map(df_cols_dict)
        sat_col = [x for x in df.columns if 'S' in x][0]
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df[df[sat_col].notnull()]
        rel_perm_dict[key]['df'] = df
    return rel_perm_dict


def relperm_parts_in_model(model, title=None, related_uuid=None):
    """Returns list of part names within model that are representing RelPerm dataframe support objects.

    arguments:
      model (model.Model): the model to be inspected for dataframes
      title (str, optional): if present, only parts with a citation title exactly matching will be
          included
      related_uuid (uuid, optional): if present, only parts relating to this uuid are included

    returns:
      list of str, each element in the list is a part name, within model, which is representing the
      support for a RelPerm object
    """

    df_parts_list = model.parts(obj_type='Grid2dRepresentation',
                                title=title,
                                extra={'relperm_table': 'true'},
                                related_uuid=related_uuid)

    return df_parts_list
=======
"""relperm.py: class for storing and retrieving dataframes of relative permeability
data as RESQML objects.

   note that this module uses the obj_Grid2dRepresentation class in a way that was not envisaged
   when the RESQML standard was defined; software that does not use resqpy is unlikely to be
   able to do much with data stored in this way
"""
import numpy as np
import pandas as pd
from scipy import interpolate
import os
import logging
import resqpy.olio.xml_et as rqet
from resqpy.olio.dataframe import DataFrame

version = '27th July 2021'

log = logging.getLogger(__name__)
log.debug(f'dataframe.py version {version}')


class RelPerm(DataFrame):
    """Class for storing and retrieving a pandas dataframe of relative permeability data.

    note:
       inherits from DataFrame class
    """

    def __init__(
            self,
            model,
            uuid=None,
            df=None,
            uom_list=None,
            realization=None,
            phase_combo=None,
            low_sal=False,
            table_index=None,
            title='relperm_table',
            column_lookup_uuid=None,
            uom_lookup_uuid=None):
        """Create a new RelPerm object from either a previously stored property or a pandas dataframe.

        arguments:
           phase_combo (str, optional): the combination of phases whose relative
           permeability behaviour is described. Options include 'water-oil', 'gas-oil' and
           'gas-water'
           low_sal (boolean, optional): if True, indicates that the water-oil table contains
           the low-salinity data for relative permeability and capillary pressure
           table_index (int, optional): the index of the relative permeability
           table when multiple relative permeability tables are present. Note, indices should start at 1.

        note:
           see DataFrame class docstring for details of other arguments
        """

        # check that either a uuid OR dataframe has been provided
        assert uuid is not None or df is not None, 'either a uuid or a dataframe must be provided'

        # check that 'phase_combo' parameter is valid
        processed_phase_combo = set([x.strip() for x in str(phase_combo).split('-')])
        assert processed_phase_combo in [{'water', 'oil'}, {'gas', 'oil'}, {'gas', 'water'},
                                         {'None'}], 'invalid phase_combo provided'

        # check that table_index is >= 1
        if table_index is not None:
            assert table_index >= 1, 'table_index cannot be less than 1'

        # check that the column names and order are as expected
        if df is not None:
            df.columns = [x.capitalize() for x in df.columns]
            if 'Pc' in df.columns:
                assert df.columns[-1] == 'Pc', 'capillary pressure data should be in the last column of the dataframe'
            if phase_combo is not None:
                if processed_phase_combo == {'water', 'oil'}:
                    expected_cols = {'Sw', 'So', 'Krw', 'Kro', 'Pc'}
                    sat_cols = {'Sw', 'So'}
                    assert df.columns[0] in sat_cols and len(set(df.columns).intersection(
                        sat_cols)) == 1, 'incorrect saturation column name and/or multiple saturation columns exist'
                    assert set(df.columns).issubset(
                        expected_cols), f'incorrect column name(s) {set(df.columns).difference(expected_cols)} \
                        in water-oil rel. perm table'
                elif processed_phase_combo == {'gas', 'oil'}:
                    expected_cols = {'Sg', 'So', 'Krg', 'Kro', 'Pc'}
                    sat_cols = {'Sg', 'So'}
                    assert df.columns[0] in sat_cols and len(set(df.columns).intersection(
                        sat_cols)) == 1, 'incorrect saturation column name and/or multiple saturation columns exist'
                    assert set(df.columns).issubset(
                        expected_cols), f'incorrect column name(s) {set(df.columns).difference(expected_cols)} \
                         in gas-oil rel. perm table'
                elif processed_phase_combo == {'gas', 'water'}:
                    expected_cols = {'Sg', 'Sw', 'Krg', 'Krw', 'Pc'}
                    sat_cols = {'Sg', 'Sw'}
                    assert df.columns[0] in sat_cols and len(set(df.columns).intersection(
                        sat_cols)) == 1, 'incorrect saturation column name and/or multiple saturation columns exist'
                    assert set(df.columns).issubset(
                        expected_cols), f'incorrect column name(s) {set(df.columns).difference(expected_cols)} \
                         in gas-oil rel. perm table'
            elif phase_combo is None:
                assert df.columns[0] in ['Sw', 'Sg', 'So'] and len(set(df.columns).intersection({'Sw', 'Sg',
                                                                                                 'So'})) == 1, \
                    'incorrect saturation column name and/or multiple saturation columns exist'
                if set(df.columns).issubset({'Sw', 'So', 'Krw', 'Kro', 'Pc'}) and len(set(df.columns)) >= 3:
                    phase_combo = 'water-oil'
                elif set(df.columns).issubset({'Sg', 'So', 'Krg', 'Kro', 'Pc'}) and len(set(df.columns)) >= 3:
                    phase_combo = 'gas-oil'
                elif set(df.columns).issubset({'Sg', 'Sw', 'Krg', 'Krw', 'Pc'}) and len(set(df.columns)) >= 3:
                    phase_combo = 'gas-water'
                else:
                    raise Exception('unexpected number of columns and/or column headers')

            # ensure that missing capillary pressure values are stored as np.nan
            for col in df.columns:
                if col.capitalize() == 'Pc':
                    df[col].replace('None', np.nan, inplace=True)

            # convert all values in the dataframe to numeric type
            df_cols = df.columns
            df[df_cols] = df[df_cols].apply(pd.to_numeric, errors='coerce')

            # ensure that no other column besides Pc has missing values
            for col in df.columns:
                if col.capitalize != 'Pc':
                    continue
                elif (df[col].isnull().sum() > 0) | ('None' in list(df[col])):
                    raise Exception(f'missing values found in {col} column')

            # check that Sw, Kr and Pc values are monotonic and that the Sw and Kr values are within the range 0-1
            sat_col = [x for x in df.columns if x[0] == 'S'][0]
            relp_corr_col = [x for x in df.columns if x == 'Kr' + sat_col[-1]][0]
            relp_opp_col = [x for x in df.columns if (x[0:2] == 'Kr') & (x[-1] != sat_col[-1])][0]
            assert (df[sat_col].is_monotonic and df[relp_corr_col].is_monotonic and df[
                relp_opp_col].is_monotonic_decreasing) or \
                (df[sat_col].is_monotonic_decreasing and df[relp_corr_col].is_monotonic_decreasing and
                    df[relp_opp_col].is_monotonic), f'{sat_col, relp_corr_col, relp_opp_col} combo is not monotonic'
            if 'Pc' in df.columns:
                assert df['Pc'].dropna().is_monotonic or df[
                    'Pc'].dropna().is_monotonic_decreasing, 'Pc values are not monotonic'
            for col in ['Sw', 'Sg', 'So', 'Krw', 'Krg', 'Kro']:
                if col in df.columns:
                    assert df[col].min() >= 0 and df[col].max() <= 1, f'{col} is not within the range 0-1'

        super().__init__(model,
                         uuid=uuid,
                         support_root=None,
                         df=df,
                         uom_list=uom_list,
                         realization=realization,
                         title=title,
                         column_lookup_uuid=column_lookup_uuid,
                         uom_lookup_uuid=uom_lookup_uuid)
        self.phase_combo = phase_combo
        self.low_sal = low_sal
        self.table_index = table_index

    def interpolate_point(self, saturation, kr_or_pc_col):
        """Returns a tuple of the saturation value and the corresponding
        interpolated relative permeability or capillary pressure value.

        arguments:
            saturation (float): the saturation at which the relative permeability or cap. pressure will be interpolated
            kr_or_pc_col (str): the column name of the parameter to be interpolated
        returns:
            tuple of float, the first element is the saturation and the second element is the interpolated value
           """
        df = self.df.copy()
        assert kr_or_pc_col.capitalize() in df.columns and kr_or_pc_col.capitalize() != df.columns[
            0], 'incorrect column name provided for interpolation'
        if kr_or_pc_col == 'PC':
            df = df[df['PC'].notnull()]
        sat_col = df.columns[0]
        x = df[sat_col]
        y = df[kr_or_pc_col]
        f = interpolate.interp1d(x, y, kind='linear', assume_sorted=True)
        x_new = saturation
        assert x.min() <= x_new <= x.max(), 'saturation value is outside the interpolation range'
        y_new = f(x_new)
        return saturation, y_new.item()

    def df_to_text(self, filepath, filename):
        """Creates text file from a dataframe of relative permeability and capillary pressure data.

        arguments:
            filepath (str): location where new text file is written to
            filename (str): name of the new text file
        returns:
            tuple of float, the first element is the saturation and the second element is the interpolated value
           """
        df = self.df.copy()
        ascii_file = os.path.join(filepath, filename + '.dat')
        df.columns = map(str.upper, df.columns)
        if {'KRW', 'KRO'}.issubset(set(df.columns)):
            df_cols_dict = {'SW': 'SW', 'KRW': 'KRW', 'KRO': 'KROW', 'PC': 'PCWO'}
            if self.low_sal:
                table_name_keyword = 'WOTABLE (LOW_SAL)\n'
            else:
                table_name_keyword = 'WOTABLE\n'
            df.columns = df.map(df_cols_dict)
        elif {'KRG', 'KRO'}.issubset(set(df.columns)):
            table_name_keyword = 'GOTABLE\n'
            df_cols_dict = {'SG': 'SG', 'KRG': 'KRG', 'KRO': 'KROG', 'PC': 'PCGO'}
            df.columns = df.map(df_cols_dict)
        elif {'KRW', 'KRW'}.issubset(set(df.columns)):
            table_name_keyword = 'GWTABLE\n'
            df_cols_dict = {'SG': 'SG', 'KRG': 'KRG', 'KRW': 'KRWG', 'PC': 'PCGW'}
            df.columns = df.map(df_cols_dict)
        else:
            raise Exception('incorrect rel. perm. column combination encountered')

        if filename + '.dat' not in os.listdir(filepath):
            with open(ascii_file, 'w') as f:
                f.write(table_name_keyword)
                df_str = df.to_string(na_rep='', index=False)
                f.write(df_str)
                f.close()
                print(f'Created new DAT file: {filename} at {filepath}')
        else:
            with open(ascii_file, 'a') as f:
                f.write('\n\n')
                f.write(table_name_keyword)
                df_str = df.to_string(na_rep='', index=False)
                f.write(df_str)
                f.close()
                print(f'Appended to DAT file: {filename} at {filepath}')

    def write_hdf5_and_create_xml(self):
        """Write relative permeability table data to hdf5 file and create xml for RESQML
        objects to represent dataframe."""

        super().write_hdf5_and_create_xml()
        # note: time series xml must be created before calling this method
        mesh_root = self.mesh.root
        # create an xml of extra metadata to indicate that this is a relative permeability table
        rqet.create_metadata_xml(mesh_root, {'relperm_table': 'true'})


def text_to_relperm_dict(filepath):
    """
    Returns a dictionary that contains dataframes with relative permeability and capillary pressure data and
    phase combinations.

    arguments:
    filepath (str): relative or full path of the text file to be processed

    returns:
    dict, each element in the dictionary contains a dataframe, with saturation and rel. permeability/capillary pressure
    data, and the phase combination being described
    """
    with open(filepath) as f:
        # create list of rows of the original ascii file with blank lines removed
        data = list(filter(None, [list(filter(None, x.strip('\n').split(' '))) for x in f.readlines()]))
        # remove comments
        data = [x for x in data if ('!' not in x)]

    # get indices of start of each new relperm table based on Nexus keywords
    table_start_positions = [i for i, l in enumerate(data) if
                             len({'WOTABLE', 'GOTABLE', 'GWTABLE'}.intersection(set(l))) == 1]
    df_cols_dict = {'SW': 'Sw', 'SG': 'Sg', 'KRW': 'Krw', 'KRG': 'Krg', 'KROW': 'Kro',
                    'KROG': 'Kro', 'KRWG': 'Krw', 'PCWO': 'Pc', 'PCGO': 'Pc', 'PCGW': 'Pc'}
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
        table_rows = data[l + 2: table_end]
        df = pd.DataFrame(table_rows, columns=table_cols)
        df.columns = df.columns.map(df_cols_dict)
        sat_col = [x for x in df.columns if 'S' in x][0]
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df[df[sat_col].notnull()]
        rel_perm_dict[key]['df'] = df
    return rel_perm_dict


def relperm_parts_in_model(model, title=None, related_uuid=None):
    """Returns list of part names within model that are representing RelPerm dataframe support objects.

    arguments:
      model (model.Model): the model to be inspected for dataframes
      title (str, optional): if present, only parts with a citation title exactly matching will be
          included
      related_uuid (uuid, optional): if present, only parts relating to this uuid are included

    returns:
      list of str, each element in the list is a part name, within model, which is representing the
      support for a RelPerm object
    """

    df_parts_list = model.parts(obj_type='Grid2dRepresentation',
                                title=title,
                                extra={'relperm_table': 'true'},
                                related_uuid=related_uuid)

    return df_parts_list
>>>>>>> 62773c094c4b6681bf7ee68f08cad5fd8f4d1e50
