"""Load a BFF 2-tape bytearray, run, save a binary trace."""

import bff_interpreter as bff
import os
import sys

if len(sys.argv) != 3:
    print("Usage: python3 save_bff_trace.py <input_path> <output_path>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

if not os.path.exists(input_path):
    print(f"File does not exist at path: {input_path}")
    sys.exit(1)

try:
    with open(input_path, 'rb') as f:
        program_bytes = f.read()
except Exception as e:
    print(f"Exception occurred: {e}")
    sys.exit(1)

too_many_iterations = 128 * 1024
bff.evaluate_and_save(bytearray(program_bytes), output_path, too_many_iterations, True)
