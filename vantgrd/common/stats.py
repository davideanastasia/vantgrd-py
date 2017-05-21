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

from collections import Counter


def compute_class_weight(y):
    c = Counter(y)

    # n_samples = y.size
    # return {0.0: float(c[1.0]) / n_samples, 1.0: float(c[0.0]) / n_samples}
    return {0: float(c[1.0]) / c[0.0], 1: 1.0}
