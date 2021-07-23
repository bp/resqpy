# test module for the derived RelPerm class located in resqpy.olio.dataframe.py

import unittest
import numpy as np
import os
import pandas as pd
import shutil, tempfile
import resqpy.model as rq
from resqpy.olio.dataframe import RelPerm, text_to_relperm_dict
# relperm_parts_in_model

class TestRelPerm(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.epc = os.path.join(self.test_dir, 'relperm.epc')   
        self.model = rq.new_model(epc_file = self.epc) 

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)
        
    def test_col_headers(self):  
        np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan],
                          [0.12, 0.065, 0.683, np.nan], [0.25, 0.205, 0.35, np.nan],
                          [0.45, 0.55, 0.018, np.nan], [0.74, 1.0, 0.0, 1e-06]])
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
        
        with self.assertRaises(AssertionError) as excval1:
            RelPerm(model = self.model, df = df1, phase_combo = phase_combo1)   
            assert str(excval1.value) ==  "incorrect saturation column name and/or multiple saturation columns exist"
            
        with self.assertRaises(AssertionError) as excval2:
            RelPerm(model = self.model, df = df2, phase_combo = phase_combo1)   
            assert str(excval2.value) ==  "incorrect column name(s) {'Krw'} in gas-oil rel. perm table"
            
        with self.assertRaises(AssertionError) as excval3:
            RelPerm(model = self.model, df = df3, phase_combo= phase_combo1)   
            assert str(excval3.value) ==  "capillary pressure data should be in the last column of the dataframe"
        
        relperm_obj = RelPerm(model = self.model, df = df4, phase_combo= phase_combo2)
        self.assertTrue(relperm_obj.phase_combo == 'gas-oil')
        
    def test_missing_vals(self): 
        np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan],
                          [0.12, np.nan, 0.683, np.nan], [0.25, 0.205, 0.35, np.nan],
                          [0.45, 0.55, 0.018, np.nan], [0.74, 1.0, np.nan, 1e-06]])
        df_cols = ['Sg', 'Krg', 'Kro', 'Pc']
        df = pd.DataFrame(np_df, columns = df_cols)
        phase_combo = 'gas-oil'
        
        with self.assertRaises(Exception) as excval:
            RelPerm(model = self.model, df = df, phase_combo = phase_combo)   
            assert str(excval.value) ==  "missing values found in Krg column"            
            assert str(excval.value) ==  "missing values found in Kro column"
            
    def test_monotonicity(self):
        np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan],
                          [0.12, 0.065, 0.683, np.nan], [0.25, 0.205, 0.35, np.nan],
                          [0.85, 0.55, 0.018, 1e-06], [0.74, 1.0, 0.0, 1e-07]])
        df_cols = ['Sg', 'Krg', 'Kro', 'Pc']
        df = pd.DataFrame(np_df, columns = df_cols)
        phase_combo = 'gas-oil'
        
        with self.assertRaises(Exception) as excval:
            RelPerm(model = self.model, df = df, phase_combo = phase_combo)   
            assert str(excval.value) ==  "Sg, Krg, Kro combo is not monotonic"      
            
        df['Sg'] = [0.0, 0.04, 0.12, 0.25, 0.45, 0.74]
        
        with self.assertRaises(Exception) as excval1:
            RelPerm(model = self.model, df = df, phase_combo = phase_combo)   
            assert str(excval1.value) ==  "Pc values are not monotonic" 
            
    def test_range(self):
        np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan],
                          [0.12, 0.065, 0.683, np.nan], [0.25, 0.205, 0.35, np.nan],
                          [0.45, 0.55, 0.018, np.nan], [0.74, 1.1, -0.001, 1e-06]])
        df_cols = ['Sg', 'Krg', 'Kro', 'Pc']
        df = pd.DataFrame(np_df, columns = df_cols)
        phase_combo = 'gas-oil'
        
        with self.assertRaises(Exception) as excval:
            RelPerm(model = self.model, df = df, phase_combo = phase_combo)   
            assert str(excval.value) ==  "Krg is not within the range 0-1"      
            assert str(excval.value) ==  "Kro is not within the range 0-1" 
    
    def test_relperm(self):
        np_df = np.array([[0.0, 0.0, 1.0, 0], [0.04, 0.015, 0.87, np.nan],
                          [0.12, 0.065, 0.689, np.nan], [0.25, 0.205, 0.35, np.nan],
                          [0.45, 0.55, 0.019, np.nan], [0.74, 1.0, 0.0, 0.000001]])
        df_cols1 = ['Sw', 'Krw', 'Kro', 'Pc']
        df1 = pd.DataFrame(np_df, columns = df_cols1)
        phase_combo1 = 'oil-water'
        low_sal = True
        uom_list = ['Euc', 'Euc', 'Euc', 'psi']
        df_cols2 = ['Sg', 'Krg', 'Kro', 'Pc']
        df2 = pd.DataFrame(np_df, columns = df_cols2)
        phase_combo2 = 'gas-oil'
        dataframe1 = RelPerm(model = self.model, df = df1, uom_list = uom_list , phase_combo = phase_combo1, low_sal = low_sal, title = 'table1')   
        dataframe2 = RelPerm(model = self.model, df = df2, uom_list = uom_list , phase_combo = phase_combo2, low_sal = low_sal, title = 'table2')
        assert dataframe1.n_cols == 4
        assert dataframe1.n_rows == 6
        assert all(dataframe1.dataframe() == df1)
        assert all(dataframe1.dataframe().columns == df_cols1)
        assert round(dataframe1.interpolate_point(saturation = 0.55, kr_or_pc_col = 'Kro')[1], 3) == 0.012
        dataframe1.write_hdf5_and_create_xml()
        dataframe2.write_hdf5_and_create_xml()
        # assert self.model.parts(extra = {'relperm_table': 'true'}) == relperm_parts_in_model(self.model)
        dataframe1.df_to_text(filepath = self.test_dir, filename = 'oil_water_test_table')
        df1_reconstructed = text_to_relperm_dict(os.path.join(self.test_dir, 'oil_water_test_table.dat'))['relperm_table1']['df']
        # assert df1.equals(df1_reconstructed)
        df1_reconstructed.iloc[3]['Kro'] == 0.689
if __name__ == '__main__':
    unittest.main()

