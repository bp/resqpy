import pandas as pd
import pytest
from resqpy.olio.read_nexus_fault import load_nexus_fault_mult_table


@pytest.mark.parametrize("file_contents, expected_fault_definition", [
    ("""MULT TX ALL PLUS MULT
FNAME BLUE
  12   3  56  56 1 60   1.0000
  14  15  55  55 1 60   0.0000

MULT TY ALL MINUS MULT
FNAME RED
  20  27  55  55 1 60   1.0000
  28  37  54  54 1 60   1.0000
                         """,
     pd.DataFrame({
         'i1': [12, 14, 20, 28],
         'i2': [3, 15, 27, 37],
         'j1': [56, 55, 55, 54],
         'j2': [56, 55, 55, 54],
         'k1': [1, 1, 1, 1],
         'k2': [60, 60, 60, 60],
         'mult': [1.0, 0.0, 1.0, 1.0],
         'grid': ['ROOT', 'ROOT', 'ROOT', 'ROOT'],
         'name': ['BLUE', 'BLUE', 'RED', 'RED'],
         'face': ['I', 'I', 'J-', 'J-']
     })),
    ("""MULT TXF ALL MINUS MULT
                                  GRID ROOT
                                  FNAME fault1
                                  22 222 90 90 1 1 1
                                  22 222 90 90 9 9 1
                                  22 222 90 90 11 11 1
                                  22 223 91 90 13 13 1
                                  22 223 91 90 14 14 1
                                  22 223 91 90 15 15 1""",
     pd.DataFrame({'i1': [22] * 6,
                   'i2': [222, 222, 222, 223, 223,
                          223],
                   'j1': [90, 90, 90, 91, 91, 91],
                   'j2': [90, 90, 90, 90, 90, 90],
                   'k1': [1, 9, 11, 13, 14, 15],
                   'k2': [1, 9, 11, 13, 14, 15],
                   'mult': [1.0] * 6,
                   'grid': ['ROOT'] * 6,
                   'name': ['fault1'] * 6,
                   'face': ['TXF'] * 6,
                   }))
],
                         ids=['basic', 'fracture_faults'])
def test_load_nexus_fault_mult_table(mocker, file_contents, expected_fault_definition):
    # Arrange
    # mock out open to return test file contents
    open_mock = mocker.mock_open(read_data=file_contents)
    mocker.patch("builtins.open", open_mock)

    # Act
    fault_mult_table = load_nexus_fault_mult_table('test/fault/file.inc')

    # Assert
    pd.testing.assert_frame_equal(fault_mult_table, expected_fault_definition)
