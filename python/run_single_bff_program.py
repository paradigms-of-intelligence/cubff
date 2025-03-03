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

from bin import cubff
import sys

args = sys.argv

if len(args) != 3:
    printf(f"Usage: python3 run_single_bff_program.py <filename> <max_steps_to_run>")
    sys.exit(1)


try:
    filename = args[1]
    with open(filename, 'rb') as f:
        program_bytes = f.read()
    num_steps = int(args[2])

except Exception as e:
    print(f"Error: {e}")

language = cubff.GetLanguage("bff")
cubff.ResetColors()

program = cubff.VectorUint8(program_bytes)
language.PrintProgram(128, program, [64])
language.RunSingleParsedProgram(program, num_steps, True)
