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

grid_width = int(sys.argv[1])
grid_height = int(sys.argv[2]) if len(sys.argv) >= 3 else grid_width

for i in range(grid_height):
    for j in range(grid_width):
        for ii in range(-RANGE, RANGE+1):
            for jj in range(-RANGE, RANGE+1):
                iii = i + ii
                jjj = j + jj
                if iii >= 0 and iii < grid_height and jjj >= 0 and jjj < grid_width and (i != iii or j != jjj):
                    print(i*grid_width+j, iii*grid_width+jjj)
