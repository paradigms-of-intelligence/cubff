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


STATES_TO_KEEP = 128
THRESHOLD_SCORE = 48

states = []
cur_index = -1


def callback(state):
    global cur_index, states
    print(f"collecting states: {state.epoch:5}", end="\r")
    sys.stdout.flush()
    states.append(state)
    if len(states) > STATES_TO_KEEP:
        states = states[1:]

    for i, score in enumerate(state.replication_per_prog):
        if score >= THRESHOLD_SCORE:
            print(i, score)
            cur_index = i
            return True

    return False


params = cubff.SimulationParams()
params.num_programs = 131072
params.callback_interval = 1
params.eval_selfrep = True
params.seed = 0

if len(sys.argv) > 1:
    params.seed = int(sys.argv[1])
    print(f"Seed = {params.seed}")

language = cubff.GetLanguage("bff")
cubff.ResetColors()

language.RunSimulation(params, None, callback)
print("states collected")

command_string = """
Commands:
- l/r to go to the previous epoch focusing on the left/right program
- "a number to run the current program pair for that number of steps (or until termination)
- s <filename> to save the current program pair to a file
"""

print(command_string)

cur_epoch = len(states)-1  # index in `states`.


def get_prog(epoch, idx):
    return states[epoch - 1].soup[idx * 64:(idx+1)*64]


program = get_prog(len(states), cur_index)
language.PrintProgram(128, program, [64])

while cur_epoch > 1:
    index = 0
    while states[cur_epoch].shuffle_idx[index] != cur_index:
        index += 1
    if index % 2 == 0:
        left = cur_index
        right = states[cur_epoch].shuffle_idx[index+1]
    else:
        right = cur_index
        left = states[cur_epoch].shuffle_idx[index-1]

    real_epoch = states[cur_epoch].epoch
    print(f"left: {left:5} right: {right:5} epoch: {real_epoch:5}", )

    program_bytes = (bytes(get_prog(cur_epoch, left)) +
                     bytes(get_prog(cur_epoch, right)))
    program = cubff.VectorUint8(program_bytes)

    language.PrintProgram(128, program, [64])

    while True:
        print("command: ", end="")
        command = input()
        if command.strip() == "l":
            cur_index = left
            cur_epoch -= 1
        elif command.strip() == "r":
            cur_index = right
            cur_epoch -= 1
        elif command[0] == "s":
            try:
                filename = command.split(maxsplit=1)[1]
                with open(filename, 'wb') as f:
                    f.write(program_bytes)
                print(f"Program pair saved to '{filename}'")
            except Exception as e:
                print(f"Error: {e}")
        else:
            try:
                command = int(command.strip())
                language.RunSingleParsedProgram(program, command, True)
            except:
                print(f"Unknown command {command}")
        break


print("state cache exhausted. change STATES_TO_KEEP to increase")
