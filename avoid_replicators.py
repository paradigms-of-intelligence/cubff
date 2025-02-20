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
# TODO: log various measurements (higher_entropy, various kinds of loops) at higher granularity than every switch

SEED = 8
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
    
    i = 0
    prev_start = 0
    while True:
        epoch = 0
        current_checkpoint_dir = os.path.join(CHECKPOINT_DIR, "{:04}".format(i))
        params.save_to = current_checkpoint_dir
        os.mkdir(current_checkpoint_dir)
        def callback(state):
            global epoch
            headhist = [0] * 128
            total = 0
            for i in range(len(state.soup) // 64):
                head0 = int(state.soup[64*i+0])
                head1 = int(state.soup[64*i+1])
                headhist[(128 + head0 - head1) % 128]+=1
                total+=1
            mode_diff = -1
            for (i, v) in enumerate(headhist):
                if v >= total/2:
                    mode_diff = i
            logline = "epoch={} higher_entropy={} head_dist_mode={}".format(state.epoch, state.higher_entropy, mode_diff)
            print(logline + "                     ", end='\r')
            print(logline, file=logfile, flush=True)
            epoch = state.epoch
            return state.higher_entropy > 3.0
        language.RunSimulation(params, None, callback)
        i+=1
        new_epoch = epoch - DELAY - 1
        if new_epoch < prev_start:
            # TODO: don't
            new_epoch = prev_start
        print("")
        logline="epoch={} seed switch, start epoch={}, diff={}".format(epoch, new_epoch, new_epoch - prev_start)
        print(logline + "                              ")
        print("> " + logline, file=logfile, flush=True)
        prev_start = new_epoch
        params.seed += 1
        params.load_from = os.path.join(current_checkpoint_dir, "{:010}.dat".format(new_epoch))

