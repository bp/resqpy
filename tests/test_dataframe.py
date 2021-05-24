# resqpy dataframe storage and retrieval test

import os
import numpy as np
import pandas as pd

import resqpy.model as rq
import resqpy.time_series as rqts
import resqpy.olio.dataframe as rqdf


def test_dataframe(tmp_path):

   epc = os.path.join(tmp_path, 'dataframe.epc')

   model = rq.Model(epc_file = epc, new_epc = True, create_basics = True, create_hdf5_ext = True)

   np_df = np.array([[0.0, 0.0, 0.0],
                  [1000.0, 0.0, 0.0],
                  [1900.0, 100.0, 0.0],
                  [2500.0, 500.0, 800.0],
                  [3200.0, 1500.0, 1600.0]])
   df_cols = ['COP', 'CWP', 'CWI']
   col_units = ['m3', 'ft3', 'm3']
   ts_list = ['2021-04-22', '2021-04-23', '2021-04-24', '2021-04-25', '2021-04-26']

   df = pd.DataFrame(np_df, columns = df_cols)

   vanilla = rqdf.DataFrame(model, df = df, title = 'vanilla')
   vanilla.write_hdf5_and_create_xml()

   ts = rqts.time_series_from_list(ts_list)
   ts.set_model(model)
   ts.create_xml()

   timetable = rqdf.TimeTable(model, df = df, title = 'time table', time_series = ts)
   timetable.write_hdf5_and_create_xml()

   propertied = rqdf.TimeTable(model, df = df, realization = 0, title = 'time table realisation',
                               time_series = ts)
   propertied.write_hdf5_and_create_xml()

   united = rqdf.TimeTable(model, df = df, realization = 0, title = 'united realisation',
                           uom_list = col_units, time_series = ts)
   united.write_hdf5_and_create_xml()

   model.store_epc()
   model.h5_release()
   del model

   model = rq.Model(epc)
   df_roots = model.roots(obj_type = 'Grid2dRepresentation', extra = {'dataframe': 'true'})
   assert len(df_roots) == 4

   v_root = model.root(obj_type = 'Grid2dRepresentation', extra = {'dataframe': 'true'}, title = 'vanilla')
   dataframe = rqdf.DataFrame(model, support_root = v_root)
   assert dataframe.n_cols == 3
   assert dataframe.n_rows == 5
   assert all(dataframe.dataframe() == df)
   assert all(dataframe.dataframe().columns == df_cols)

   tt_root = model.root(obj_type = 'Grid2dRepresentation', extra = {'dataframe': 'true'}, title = 'time table')
   time_table = rqdf.TimeTable(model, support_root = tt_root)
   assert time_table.time_series().timestamps == ts_list
   assert all(time_table.dataframe() == df)

   u_root = model.root(obj_type = 'Grid2dRepresentation', extra = {'dataframe': 'true'}, title = 'united realisation')
   ur = rqdf.TimeTable(model, support_root = u_root)
   assert ur.uom_list == col_units
   assert ur.time_series().timestamps == ts_list

   # only dataframes with a realization number are stored as a property object
   assert len(model.parts(obj_type = 'ContinuousProperty')) == 2
