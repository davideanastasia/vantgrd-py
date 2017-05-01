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
from collections import Counter


def compute_class_weight(y):
    c = Counter(y)

    # n_samples = y.size
    # return {0.0: float(c[1.0]) / n_samples, 1.0: float(c[0.0]) / n_samples}
    return {0: float(c[1.0]) / c[0.0], 1: 1.0}
