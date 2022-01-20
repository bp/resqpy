# test module for the derived RelPerm class located in resqpy.olio.dataframe.py

import os

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import resqpy.model as rq
from resqpy.olio.relperm import (RelPerm, relperm_parts_in_model, text_to_relperm_dict)


def test_col_headers(tmp_path):
    epc = os.path.join(tmp_path, 'model.epc')
    model = rq.new_model(epc)
    np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan], [0.12, 0.065, 0.683, np.nan],
                      [0.25, 0.205, 0.35, np.nan], [0.45, 0.55, 0.018, np.nan], [0.74, 1.0, 0.0, 1e-06]])
    df_cols1 = ['Sw', 'Krg', 'Kro', 'Pc']
    df_cols2 = ['Sg', 'Krg', 'Krw', 'Pc']
    df_cols3 = ['Sg', 'Krg', 'Pc', 'Kro']
    df_cols4 = ['Sg', 'krg', 'Kro', 'pC']
    df1 = pd.DataFrame(np_df, columns = df_cols1)
    df2 = pd.DataFrame(np_df, columns = df_cols2)
    df3 = pd.DataFrame(np_df, columns = df_cols3)
    df4 = pd.DataFrame(np_df, columns = df_cols4)
    phase_combo1 = 'gas-oil'
    phase_combo2 = None

    with pytest.raises(ValueError) as excval1:
        RelPerm(model = model, df = df1, phase_combo = phase_combo1)
    assert "incorrect saturation column name and/or multiple saturation columns exist" in str(excval1.value)

    with pytest.raises(ValueError) as excval2:
        RelPerm(model = model, df = df2, phase_combo = phase_combo1)
    assert "incorrect column name(s) {'Krw'}" in str(excval2.value)

    with pytest.raises(ValueError) as excval3:
        RelPerm(model = model, df = df3, phase_combo = phase_combo1)
    assert "capillary pressure data should be in the last column of the dataframe" in str(excval3.value)

    relperm_obj = RelPerm(model = model, df = df4, phase_combo = phase_combo2)
    assert relperm_obj.phase_combo == 'gas-oil'


def test_missing_vals(tmp_path):
    epc = os.path.join(tmp_path, 'model.epc')
    model = rq.new_model(epc)
    np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan], [0.12, np.nan, 0.683, np.nan],
                      [0.25, 0.205, 0.35, np.nan], [0.45, 0.55, 0.018, np.nan], [0.74, 1.0, np.nan, 1e-06]])
    df_cols = ['Sg', 'Krg', 'Kro', 'Pc']
    df = pd.DataFrame(np_df, columns = df_cols)
    phase_combo = 'gas-oil'

    with pytest.raises(Exception) as excval:
        RelPerm(model = model, df = df, phase_combo = phase_combo)
    assert "missing values found in Krg column" in str(excval.value)


def test_monotonicity(tmp_path):
    epc = os.path.join(tmp_path, 'model.epc')
    model = rq.new_model(epc)
    np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan], [0.12, 0.065, 0.683, np.nan],
                      [0.25, 0.205, 0.35, np.nan], [0.85, 0.55, 0.018, 1e-06], [0.74, 1.0, 0.0, 1e-07]])
    df_cols = ['Sg', 'Krg', 'Kro', 'Pc']
    df = pd.DataFrame(np_df, columns = df_cols)
    phase_combo = 'gas-oil'

    with pytest.raises(ValueError) as excval:
        RelPerm(model = model, df = df, phase_combo = phase_combo)
    assert "('Sg', 'Krg', 'Kro') combo is not monotonic" in str(excval.value)

    df['Sg'] = [0.0, 0.04, 0.12, 0.25, 0.45, 0.74]
    with pytest.raises(ValueError) as excval1:
        RelPerm(model = model, df = df, phase_combo = phase_combo)
    assert "Pc values are not monotonic" in str(excval1.value)


def test_range(tmp_path):
    epc = os.path.join(tmp_path, 'model.epc')
    model = rq.new_model(epc)
    np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan], [0.12, 0.065, 0.683, np.nan],
                      [0.25, 0.205, 0.35, np.nan], [0.45, 0.55, 0.018, np.nan], [0.74, 1.1, -0.001, 1e-06]])
    df_cols = ['Sg', 'Krg', 'Kro', 'Pc']
    df = pd.DataFrame(np_df, columns = df_cols)
    phase_combo = 'gas-oil'

    with pytest.raises(ValueError) as excval:
        RelPerm(model = model, df = df, phase_combo = phase_combo)
    assert "Krg is not within the range 0-1" in str(excval.value)


def test_relperm(tmp_path):
    epc = os.path.join(tmp_path, 'model.epc')
    model = rq.new_model(epc)
    np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan], [0.12, 0.065, 0.689, np.nan],
                      [0.25, 0.205, 0.35, np.nan], [0.45, 0.55, 0.019, np.nan], [0.74, 1.0, 0.0, 0.000001]])
    df_cols1 = ['Sw', 'Krw', 'Kro', 'Pc']
    df1 = pd.DataFrame(np_df, columns = df_cols1)
    phase_combo1 = 'oil-water'
    uom_list = ['Euc', 'Euc', 'Euc', 'psi']
    df_cols2 = ['Sg', 'Krg', 'Kro', 'Pc']
    df2 = pd.DataFrame(np_df, columns = df_cols2)
    phase_combo2 = 'gas-oil'
    dataframe1 = RelPerm(model = model,
                         df = df1,
                         uom_list = uom_list,
                         phase_combo = phase_combo1,
                         low_sal = True,
                         table_index = 1,
                         title = 'table1')
    dataframe2 = RelPerm(model = model,
                         df = df2,
                         uom_list = uom_list,
                         phase_combo = phase_combo2,
                         low_sal = False,
                         title = 'table2')
    assert dataframe1.n_cols == 4
    assert dataframe1.n_rows == 6
    assert_frame_equal(dataframe1.dataframe(), df1)
    assert all(dataframe1.dataframe().columns == df_cols1)
    assert round(dataframe1.interpolate_point(saturation = 0.55, kr_or_pc_col = 'Kro')[1], 3) == 0.012
    dataframe1.write_hdf5_and_create_xml()
    dataframe2.write_hdf5_and_create_xml()
    assert model.part(extra = {'relperm_table': 'true', 'low_sal': 'true'}) == model.part(title = 'table1')
    assert len(relperm_parts_in_model(model, low_sal = False)) == 1
    dataframe1.df_to_text(filepath = tmp_path, filename = 'oil_water_test_table')
    # reconstruct a dataframe of rel. perm. data from a text file
    df1_reconstructed_from_file = text_to_relperm_dict(os.path.join(tmp_path,
                                                                    'oil_water_test_table.dat'))['relperm_table1']['df']
    assert_frame_equal(df1, df1_reconstructed_from_file)
    assert df1_reconstructed_from_file.iloc[3]['Kro'] == 0.350
    # reconstruct a dataframe of rel. perm. data from a string
    with open(os.path.join(tmp_path, 'oil_water_test_table.dat')) as f:
        relperm_string = f.read()
    df1_reconstructed_from_string = text_to_relperm_dict(relperm_string, is_file = False)['relperm_table1']['df']
    assert_frame_equal(df1, df1_reconstructed_from_string)
    # initialize a RelPerm object from an existing uuid
    new_wo_obj = RelPerm(model, uuid = model.uuid(title = 'table1'))
    assert new_wo_obj.phase_combo == phase_combo1
    assert new_wo_obj.low_sal == 'true'
    assert new_wo_obj.table_index == '1'
    assert new_wo_obj.extra_metadata == {
        'dataframe': 'true',
        'low_sal': 'true',
        'phase_combo': 'oil-water',
        'relperm_table': 'true',
        'table_index': '1'
    }


def test_relperm_df_none_uuid_none(tmp_path):
    # Arrange
    test_model = rq.new_model(os.path.join(tmp_path, 'test'))

    # Act & Assert
    with pytest.raises(ValueError) as e:
        RelPerm(test_model)
    assert "either a uuid or a dataframe must be provided" in str(e.value)


def test_relperm_invalid_phase_combo(tmp_path):
    # Arrange
    test_model = rq.new_model(os.path.join(tmp_path, 'test'))
    np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan], [0.12, 0.065, 0.689, np.nan],
                      [0.25, 0.205, 0.35, np.nan], [0.45, 0.55, 0.019, np.nan], [0.74, 1.0, 0.0, 0.000001]])
    df_cols = ['Sw', 'Krw', 'Kro', 'Pc']
    test_df = pd.DataFrame(np_df, columns = df_cols)
    test_phase_combo = 'wateroil'

    # Act & Assert
    with pytest.raises(ValueError) as e:
        RelPerm(test_model, df = test_df, phase_combo = test_phase_combo)
    assert "invalid phase_combo provided" in str(e.value)


def test_relperm_table_index_equal_0(tmp_path):
    # Arrange
    test_model = rq.new_model(os.path.join(tmp_path, 'test'))
    np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan], [0.12, 0.065, 0.689, np.nan],
                      [0.25, 0.205, 0.35, np.nan], [0.45, 0.55, 0.019, np.nan], [0.74, 1.0, 0.0, 0.000001]])
    df_cols = ['Sw', 'Krw', 'Kro', 'Pc']
    test_df = pd.DataFrame(np_df, columns = df_cols)
    test_table_index = 0

    # Act & Assert
    with pytest.raises(ValueError) as e:
        RelPerm(test_model, df = test_df, table_index = test_table_index)
    assert "table_index cannot be less than 1" in str(e.value)


@pytest.mark.parametrize("test_phase_combo", ['gas-water', 'water-gas'])
def test_relperm_gas_water_phase_combo_invalid_first_column(tmp_path, test_phase_combo):
    # Arrange
    test_model = rq.new_model(os.path.join(tmp_path, 'test'))
    np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan], [0.12, 0.065, 0.689, np.nan],
                      [0.25, 0.205, 0.35, np.nan], [0.45, 0.55, 0.019, np.nan], [0.74, 1.0, 0.0, 0.000001]])
    df_cols = ['So', 'Krw', 'Kro', 'Pc']
    test_df = pd.DataFrame(np_df, columns = df_cols)

    # Act & Assert
    with pytest.raises(ValueError) as e:
        RelPerm(test_model, df = test_df, phase_combo = test_phase_combo)
    assert "incorrect saturation column name and/or multiple saturation columns exist" in str(e.value)


@pytest.mark.parametrize("test_phase_combo", ['gas-water', 'water-gas'])
def test_relperm_gas_water_phase_combo_invalid_columns(tmp_path, test_phase_combo):
    # Arrange
    test_model = rq.new_model(os.path.join(tmp_path, 'test'))
    np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan], [0.12, 0.065, 0.689, np.nan],
                      [0.25, 0.205, 0.35, np.nan], [0.45, 0.55, 0.019, np.nan], [0.74, 1.0, 0.0, 0.000001]])
    df_cols = ['Sg', 'Krw', 'Kro', 'Pc']
    test_df = pd.DataFrame(np_df, columns = df_cols)

    # Act & Assert
    with pytest.raises(ValueError) as e:
        RelPerm(test_model, df = test_df, phase_combo = test_phase_combo)
    assert "incorrect column name(s) {'Kro'}" in str(e.value)


def test_relperm_no_phase_combo_invalid_first_column(tmp_path):
    # Arrange
    test_model = rq.new_model(os.path.join(tmp_path, 'test'))
    np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan], [0.12, 0.065, 0.689, np.nan],
                      [0.25, 0.205, 0.35, np.nan], [0.45, 0.55, 0.019, np.nan], [0.74, 1.0, 0.0, 0.000001]])
    df_cols = ['Sn', 'Krw', 'Kro', 'Pc']
    test_df = pd.DataFrame(np_df, columns = df_cols)

    # Act & Assert
    with pytest.raises(ValueError) as e:
        RelPerm(test_model, df = test_df)
    assert "incorrect saturation column name and/or multiple saturation columns exist" in str(e.value)


@pytest.mark.parametrize("test_cols,test_phase_combo", [(['Sw', 'Krw', 'Kro', 'Pc'], 'water-oil'),
                                                        (['Sg', 'Krg', 'Kro', 'Pc'], 'gas-oil'),
                                                        (['Sw', 'Krw', 'Krg', 'Pc'], 'gas-water')])
def test_relperm_no_phase_combo(tmp_path, test_cols, test_phase_combo):
    # Arrange
    test_model = rq.new_model(os.path.join(tmp_path, 'test'))
    np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan], [0.12, 0.065, 0.689, np.nan],
                      [0.25, 0.205, 0.35, np.nan], [0.45, 0.55, 0.019, np.nan], [0.74, 1.0, 0.0, 0.000001]])
    test_df = pd.DataFrame(np_df, columns = test_cols)

    # Act
    relperm = RelPerm(test_model, df = test_df)

    # Assert
    assert relperm.phase_combo == test_phase_combo
