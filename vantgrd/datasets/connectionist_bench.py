#
# This file is part of vantgrd-py. vantgrd-py is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright 2017 Davide Anastasia
#
import numpy as np


def read_connectionist_bench(filename):
    X_ = []
    y_ = []

    with open(filename) as i_file:
        for line in i_file:
            tokens = line.strip().split(',')

            X_.append([float(tokens[x]) for x in xrange(0, 60)])
            y_.append(0. if tokens[60] == 'R' else 1.)

    # return np.asarray(X_, dtype=np.float64), np.asarray(y_, dtype=np.float64)
    return np.ascontiguousarray(X_, dtype=np.float64), np.ascontiguousarray(y_, dtype=np.float64)
