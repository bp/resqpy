"""Writing faults in NEXUS format"""

version = '29th April 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import pandas as pd

import resqpy.olio.trademark as tm

#  both functions expect df to be a pandas dataframe with columns:
#   name, i1, i2, j1, j2, k1, k2, face
# where name is the fault/baffle name
# i,j,k indices are in simulator protocol (one based)
# face is either I+ or J+ for now


def write_faults_nexus(filename, df, grid_name = 'ROOT'):
    """Creates a Nexus include file holding MULT keywords with FNAME and face data, from pandas dataframe."""

    def _write_header_lines(fp, T, grid_name, fault_name):
        fp.write('\nMULT\t' + T + '\tALL\tPLUS\tMULT\n')
        fp.write('\tGRID\t' + grid_name + '\n')
        fp.write('\tFNAME\t' + fault_name + '\n')
        if len(fault_name) > 256:
            log.warning('exported fault name longer than Nexus limit of 256 characters: ' + fault_name)
            tm.log_nexus_tm('warning')

    def _write_rows(fp, df):
        for row in range(len(df)):
            fp.write('\t{0:1d}\t{1:1d}\t{2:1d}\t{3:1d}\t{4:1d}\t{5:1d}\t1.0\n'.format(
                df.iloc[row, 1], df.iloc[row, 2], df.iloc[row, 3], df.iloc[row, 4], df.iloc[row, 5], df.iloc[row, 6]))

    log.info('writing FNAME data in Nexus format to file: ' + filename)
    tm.log_nexus_tm('info')
    assert df is not None and len(df) > 0, 'no data in faults dataframe'

    with open(filename, 'w') as fp:
        fault_names = pd.unique(df.name)
        for fault_name in fault_names:
            fdf = df[df.name == fault_name]
            fdfi = fdf[fdf.face == 'I+']
            fdfj = fdf[fdf.face == 'J+']
            if len(fdfi):
                _write_header_lines(fp, 'TX', grid_name, fault_name)
                _write_rows(fp, fdfi)
            if len(fdfj):
                _write_header_lines(fp, 'TY', grid_name, fault_name)
                _write_rows(fp, fdfj)
