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

import os
import random
import sys
import re
import string
import struct

from bin import cubff

NUM_RUNS = 100

INIT_SEED = 0
THRESHOLD_ENTROPY = 3

MAX_EPOCHS = 100000


def StandardParams(f):
    params = cubff.SimulationParams()
    params.seed = INIT_SEED
    if f:
        params.load_from = f
    return params


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
        nonlocal ok
        ok = state.higher_entropy > THRESHOLD_ENTROPY
        return ok or state.epoch > MAX_EPOCHS + initial_epoch

    cubff.RunSimulation("bff", params, None, callback)

    if ok:
        return epochs
    return None


res_cond = []
res_nostart = []
seeds = list(range(NUM_RUNS))
with open("".join(random.choice(string.ascii_lowercase) for _ in range(8)), "a") as rs:
    for s in seeds:
      print(s)
      f = f"{s}.dat"
      header = struct.pack("=Q", 0) + struct.pack("=Q", 131072) + struct.pack("=Q", 0)
      random.seed(0)

      for _ in range(131072):
          p = (
              random.randrange(0, 256).to_bytes()
              + random.randrange(256).to_bytes()
              + b"["
              + b"".join([random.randrange(256).to_bytes() for _ in range(60)])
              + b"]"
          )
          header = header + p

      with open(f, "wb") as wf:
          wf.write(header)
      params_cond = StandardParams(f)
      params_cond.seed = s
      res_cond.append(find_threshold_epoch(params_cond))

      header = header[:-62]+ random.randrange(256).to_bytes()+header[-61:-1]+random.randrange(256).to_bytes()
      with open(f, "wb") as wf:
          wf.write(header)
      params_nostart = StandardParams(f)
      params_nostart.seed = s
      res_nostart.append(find_threshold_epoch(params_nostart))
      print(
          s,
          ",",
          res_cond[-1],
          ",",
          sum(res_cond) / len(res_cond),
          ",",
          res_nostart[-1],
          ",",
          sum(res_nostart) / len(res_nostart),
      )

      rs.write(f"{s},{res_cond[-1]},{res_nostart[-1]}\n")
