#! /usr/bin/env python

"""
Based on: https://gist.github.com/daniel-perry/8744618
Minor changes:
- lower case for dict keys

BIL parser to load elevation info from sites like
http://earthexplorer.usgs.gov/

Mostly based of:
http://stevendkay.wordpress.com/2010/05/29/parsing-usgs-bil-digital-elevation-models-in-python/

Documentation for the format itself:
http://webhelp.esri.com/arcgisdesktop/9.2/index.cfm?TopicName=BIL,_BIP,_and_BSQ_raster_files

Documentation for the accompanying world files:
http://webhelp.esri.com/arcgisdesktop/9.2/index.cfm?TopicName=World_files_for_raster_datasets


usage:
from bil_parser impor BilParser
bp = BilParser("filename.hdr") # expects to also find filename.bil
print bp.header
print bp.values.shape
imshow(bp.values)
"""

import os
import struct
import numpy as np


def parse_header(hdr):
    """
    Parse a BIL header .hdr file, like:

    BYTEORDER I
    LAYOUT BIL
    NROWS 1201
    NCOLS 1201
    ...
    """
    contents = open(hdr).read()
    lines = contents.strip().splitlines()
    header = {}
    for li in lines:
        key, _, value = li.partition(" ")
        header[key] = value.strip()

    return header


def parse_bil(path, rows, cols, dtype):
    # where you put the extracted BIL file
    fi = open(path, "rb")
    contents = fi.read()
    fi.close()

    # unpack binary data into a flat tuple z
    n = int(rows * cols)
    if dtype == "float":
        s = "<%df" % (n,)
    else:  # spec says to assume unsigned int if no type specified..
        s = "<%dH" % (n,)  # unsigned int
    z = struct.unpack(s, contents)

    values = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            val = z[(cols * r) + c]
            if (val == 65535 or val < 0 or val > 20000):
                # may not be needed depending on format, and the "magic number"
                # value used for 'void' or missing data
                val = 0.0
            values[r][c] = float(val)
    return values


class BilParser(object):
    def __init__(self, headerpath):
        self.basepath = os.path.splitext(headerpath)[0]
        self.header = parse_header(self.basepath + ".hdr")
        self.values = parse_bil(
            self.basepath + ".bil",
            rows=int(self.header['nrows']),
            cols=int(self.header['ncols']),
            dtype=self.header['pixeltype'])

