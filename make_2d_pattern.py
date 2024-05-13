# Copyright 2024 Google LLC
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

import sys

RANGE = 2

grid_size = int(sys.argv[1])

for i in range(grid_size):
    for j in range(grid_size):
        for ii in range(-RANGE, RANGE+1):
            for jj in range(-RANGE, RANGE+1):
                iii = i + ii
                jjj = j + jj
                if iii >= 0 and iii < 128 and jjj >= 0 and jjj < 128 and (i != iii or j != jjj):
                    print(i*128+j, iii*128+jjj)
