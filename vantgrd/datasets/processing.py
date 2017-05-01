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


def min_max_scale(X):
    return (X - X.min(0)) / X.ptp(0)


def normalize_data(X):
    return (X - X.mean(axis=0, dtype=np.float64)) / X.std(axis=0, dtype=np.float64)
