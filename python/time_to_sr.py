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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
+ '/bin/')
import cubff

NUM_RUNS = 100

INIT_SEED = 0
THRESHOLD_ENTROPY = 3

MAX_EPOCHS = 1000000


def find_threshold_epoch(params):
    initial_epoch = None
    epochs = 0
    ok = False

    def callback(state):
        nonlocal initial_epoch
        if not initial_epoch:
            initial_epoch = state.epoch
        nonlocal epochs
        epochs = state.epoch
        print(epochs,state.higher_entropy)
        nonlocal ok
        ok = state.higher_entropy > THRESHOLD_ENTROPY
        return ok or state.epoch > MAX_EPOCHS + initial_epoch

    language = cubff.GetLanguage("bff_noheads")
    language.RunSimulation(params, None, callback)

    if ok:
        return epochs
    return None


res = []
seeds = list(range(NUM_RUNS))
with open(f"../bffnoheadslog/{NUM_RUNS}.log", "a") as rs:
    for s in seeds:
        print("starting",s)
        params = cubff.SimulationParams()
        params.seed = s
        params.callback_interval = 32

        
        res.append(find_threshold_epoch(params))

        rs.write(f"{s},{res[-1]}\n")
