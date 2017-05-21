# Copyright 2017 Davide Anastasia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
