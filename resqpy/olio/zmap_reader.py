"""Functions for reading zmap and roxar format files."""

# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)

import numpy as np


def read_zmap_header(inputfile):
    """Reads header lines from a zmap format file.

    returns:
       header_lines_count, no_rows, no_cols, minx, maxx, miny, maxy, null_value
    """

    # read header, read until second '@', record header lines and content
    with open(inputfile, 'r') as infile:
        comments = []
        head = []
        a = 0  # count '@'s
        headers = 0  # count header+comment lines
        while a < 2:
            line = infile.readline()
            headers += 1
            if line[:1] == '@':
                a += 1
            if line[:1] == '!':
                comments.append(line.strip())
            elif a == 1 or (line[:1] == '@'):
                row = line.strip().strip('@').split(',')
                head.append(row)
            else:
                log.error("Header section does not seem to be defined by 2 @s")
                return None, None, None
        line = infile.readline()
        if line[0] == '+':
            headers = headers + 1  # add extra line for the + symbol..

    # for c in comments:
    #     log.debug(c)

    # ok now process the header
    # nodes_per_line                          = int(head[0][2])
    # field_w                                 = head[1][0]
    null_value = head[1][1].strip()
    null_value2 = head[1][2].strip()
    # dec_places                              = head[1][3]
    # strt_c                                  = head[1][4]
    no_rows = int(head[2][0])
    no_cols = int(head[2][1])
    minx = np.float64(head[2][2])
    maxx = np.float64(head[2][3])
    miny = np.float64(head[2][4])
    maxy = np.float64(head[2][5])

    # decide on the null value
    if not null_value:
        null_value = null_value2
    # log.debug("Read {} header lines, we have {} rows, {} cols, and data from {} to {} and {} to {}".format(
    #     headers, no_rows, no_cols, minx, maxx, miny, maxy))

    return headers, no_rows, no_cols, minx, maxx, miny, maxy, null_value


# note: the RMS text format was previously known as the Roxar format


def read_roxar_header(inputfile):
    """Reads header lines from a roxar format file.

    returns:
       header_lines_count, no_rows, no_cols, minx, maxx, miny, maxy, null_value
    """

    with open(inputfile, 'r') as fp:
        words = fp.readline().split()
        assert words[0] == '-996', 'RMS text format indicator -996 not found'
        no_rows = int(words[1])
        # dx = float(words[2])
        # dy = float(words[3])
        words = fp.readline().split()
        minx = float(words[0])
        maxx = float(words[1])
        miny = float(words[2])
        maxy = float(words[3])
        words = fp.readline().split()
        no_cols = int(words[0])
    headers = 4
    null_value = '9999900.0000'
    return headers, no_rows, no_cols, minx, maxx, miny, maxy, null_value


def read_mesh(inputfile, dtype = np.float64, format = None):
    """Reads a mesh (lattice) from a zmap or roxar format file.

    returns:
       x, y, f: each a numpy float array of shape (no_rows, no_cols)
    """

    if format == 'zmap':
        headers, no_rows, no_cols, minx, maxx, miny, maxy, null_value = read_zmap_header(inputfile)
    elif format in ['rms', 'roxar']:
        headers, no_rows, no_cols, minx, maxx, miny, maxy, null_value = read_roxar_header(inputfile)
    else:
        raise ValueError('format not recognised for read_mesh: ' + str(format))
    # load the values in, converting null value to NaN's

    infile = open(inputfile, 'r')
    indata = infile.readlines()
    infile.close()
    c = 0
    f = np.zeros(no_cols * no_rows, dtype = dtype)
    n = 0
    for line in indata:
        c += 1
        if c <= headers:
            continue
        else:
            for x in line.split():
                if (x == null_value) or (dtype(x) == dtype(null_value)):
                    f[n] = np.nan
                else:
                    f[n] = dtype(x)
                n += 1
        # if (n % 10000 == 0) and (n > 10000):
        #     log.debug("Read node {} of {}".format(n, no_cols * no_rows))

    # log.debug("Read {} nodes".format(len(f)))

    if format == 'zmap':
        # reshape it, and swap axis. Beacuse columns major order from fortran.
        f = f.reshape((no_cols, no_rows)).swapaxes(0, 1)
    else:  # format in ['rms', 'roxar']
        f = f.reshape((no_rows, no_cols))

    # now generate x and y coords
    x = np.linspace(minx, maxx, no_cols)  # get x axis coords
    if format == 'zmap':
        y = np.linspace(maxy, miny, no_rows)
    else:  # format in ['rms', 'roxar']
        y = np.linspace(miny, maxy, no_rows)
    x, y = np.meshgrid(x, y)  # get x and y of every node
    assert x.shape == y.shape == f.shape

    return x, y, f


def read_zmapplusgrid(inputfile, dtype = np.float64):
    """Read zmapplus grid (surface mesh); returns triple (x, y, z) 2D arrays."""

    return read_mesh(inputfile, dtype = dtype, format = 'zmap')


def read_roxar_mesh(inputfile, dtype = np.float64):
    """Read RMS text format surface mesh; returns triple (x, y, z) 2D arrays.

    note:
       the RMS text format was previously known as the Roxar format
    """

    return read_mesh(inputfile, dtype = dtype, format = 'rms')


def read_rms_text_mesh(inputfile, dtype = np.float64):
    """Read RMS text format surface mesh; returns triple (x, y, z) 2D arrays.

    note:
       the RMS text format was previously known as the Roxar format
    """

    return read_roxar_mesh(inputfile, dtype = dtype)
