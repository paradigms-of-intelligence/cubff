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

import os

# dir for snapshots
# somewhere to store seed changes
# do we clean up "future" in the dir?
# how far do we jump back?
# do we care how often we test?

# TODO: split checkpoints by attempts (we now overwrite overlapping epochs)

SEED = 2
DELAY = 512
LANG = "bff"

CHECKPOINT_DIR = "./avoid-{}-seed{}-delay{}".format(LANG, SEED, DELAY)

os.mkdir(CHECKPOINT_DIR)

with open(os.path.join(CHECKPOINT_DIR, "log.txt"), "w") as logfile:

    language = cubff.GetLanguage(LANG)

    params = cubff.SimulationParams()
    params.num_programs = 131072
    params.seed = SEED * 1000
    params.save_to = CHECKPOINT_DIR
    params.save_interval = params.callback_interval;
    
    prev_start = 0
    while True:
        epoch = 0
        def callback(state):
            global epoch
            print("epoch={}".format(state.epoch), end='\r')
            epoch = state.epoch
            return state.higher_entropy > 3.0
        language.RunSimulation(params, None, callback)
        new_epoch = epoch - DELAY - 1
        logline="epoch={} seed switch, start epoch={}, diff={}".format(epoch, new_epoch, new_epoch - prev_start)
        print(logline)
        print(logline, file=logfile, flush=True)
        prev_start = new_epoch
        params.seed += 1
        params.load_from = os.path.join(CHECKPOINT_DIR, "{:010}.dat".format(new_epoch))

