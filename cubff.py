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

from bin import cubff


def callback(state):
    print(state.epoch, state.brotli_size)
    state.print_program(1)
    return state.epoch > 1024


params = cubff.SimulationParams()
params.num_programs = 131072
params.seed = 0

cubff.RunSimulation("bff_noheads", params, None, callback)
