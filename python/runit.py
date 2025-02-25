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

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) + "/bin/"
)
import cubff
import json
import sys


language = cubff.GetLanguage(sys.argv[1])


def callback(state):
    print(json.dumps({"epoch": state.epoch, "higher_entropy": state.higher_entropy}))
    return state.higher_entropy > 3.0


params = cubff.SimulationParams()
params.num_programs = 131072
params.seed = int(sys.argv[2])

language.RunSimulation(params, None, callback)
