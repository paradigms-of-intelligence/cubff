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
import sys


TARGET_EPOCH = 3713
STATES_TO_KEEP = 128

states = []


def callback(state):
    print(f"collecting states: {state.epoch:5}/{TARGET_EPOCH:5}", end="\r")
    sys.stdout.flush()
    if state.epoch > TARGET_EPOCH - STATES_TO_KEEP:
        states.append(state)
    return state.epoch >= TARGET_EPOCH


params = cubff.SimulationParams()
params.num_programs = 131072
params.seed = 0
params.callback_interval = 1

language = cubff.GetLanguage("bff")
cubff.ResetColors()

language.RunSimulation(params, None, callback)
print("states collected")

print("Commands: l/r to go to the previous epoch focusing on the left/right program, a number to run the current program pair for that number of steps (or until termination)")

cur = 0
cur_epoch = len(states)-1  # index in `states`.


def get_prog(epoch, idx):
    return states[epoch - 1].soup[idx * 64:(idx+1)*64]


while cur_epoch > 1:
    index = 0
    while states[cur_epoch].shuffle_idx[index] != cur:
        index += 1
    if index % 2 == 0:
        left = cur
        right = states[cur_epoch].shuffle_idx[index+1]
    else:
        right = cur
        left = states[cur_epoch].shuffle_idx[index-1]

    real_epoch = max(TARGET_EPOCH - 128, 0) + cur_epoch
    print(f"left: {left:5} right: {right:5} epoch: {real_epoch:5}", )

    program = cubff.VectorUint8(bytes(get_prog(cur_epoch, left)) +
                                bytes(get_prog(cur_epoch, right)))

    language.PrintProgram(128, program, [64])

    while True:
        print("command: ", end="")
        command = input()
        if command.strip() == "l":
            cur = left
            cur_epoch -= 1
        elif command.strip() == "r":
            cur = right
            cur_epoch -= 1
        else:
            try:
                command = int(command.strip())
                language.RunSingleParsedProgram(program, command, True)
            except:
                print(f"Unknown command {command}")
        break


print("state cache exhausted. change STATES_TO_KEEP to increase")
