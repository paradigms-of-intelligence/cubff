# Copyright 2025 Google LLC
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

"""
Script to compare the time to selfreplication for starting with and without an inserted loop
"""

import os
import random
import sys
import re
import string
import struct

dir_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(dir_path, "..", "bin")))
import cubff

NUM_RUNS = 100

INIT_SEED = 0
THRESHOLD_ENTROPY = 3

MAX_EPOCHS = 1000000
LANG = "forthtrivial"


def find_threshold_epoch(params):
    initial_epoch = None
    epoch = 0
    ok = False

    def callback(state):
        nonlocal initial_epoch
        if not initial_epoch:
            initial_epoch = state.epoch
        nonlocal epoch
        epoch = state.epoch
        print(epoch, state.higher_entropy)
        nonlocal ok
        ok = state.higher_entropy > THRESHOLD_ENTROPY
        return ok or state.epoch > MAX_EPOCHS + initial_epoch

    language = cubff.GetLanguage(LANG)
    language.RunSimulation(params, None, callback)

    if ok:
        return epoch
    return None


res = []
seeds = list(range(NUM_RUNS))
log_dir = os.path.join(dir_path, "..", "experimentlogs", "time_to_sr")
os.makedirs(log_dir, exist_ok=True)
with open(os.path.join(log_dir, LANG + str(NUM_RUNS) + ".log"), "a") as rs:
    for s in seeds:
        print("starting", s)
        params = cubff.SimulationParams()
        params.seed = s
        params.callback_interval = 32

        res.append(find_threshold_epoch(params))

        rs.write(f"{s},{res[-1]}\n")
