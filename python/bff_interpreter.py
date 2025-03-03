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

"""Pure Python BFF interpreter."""

from enum import Enum
import array
from typing import List, Tuple, Optional, Dict, Callable, Any
from colorama import Fore, Back, Style, init
import struct

# Initialize colorama
init()

# Constant for tape size
SINGLE_TAPE_SIZE = 64


class BffOp(Enum):
    LOOP_START = 0
    LOOP_END = 1
    PLUS = 2
    MINUS = 3
    COPY01 = 4
    COPY10 = 5
    DEC0 = 6
    INC0 = 7
    DEC1 = 8
    INC1 = 9
    NULL = 10
    NOOP = 11


def name() -> str:
    return "BFF"


def get_op_kind(c: int) -> BffOp:
    if c == ord('['):
        return BffOp.LOOP_START
    elif c == ord(']'):
        return BffOp.LOOP_END
    elif c == ord('+'):
        return BffOp.PLUS
    elif c == ord('-'):
        return BffOp.MINUS
    elif c == ord('.'):
        return BffOp.COPY01
    elif c == ord(','):
        return BffOp.COPY10
    elif c == ord('<'):
        return BffOp.DEC0
    elif c == ord('>'):
        return BffOp.INC0
    elif c == ord('{'):
        return BffOp.DEC1
    elif c == ord('}'):
        return BffOp.INC1
    elif c == 0:
        return BffOp.NULL
    else:
        return BffOp.NOOP


def command_repr() -> str:
    return "[]+-.,<>{}"


def character_repr() -> List[str]:
    """Create a mapping of byte values to their string representations.

    This replicates the static array of 256 Unicode strings from the C++ code.
    """
    # Initialize all values with their default Unicode mappings
    data = [chr(0x100 + i) for i in range(256)]

    # Special letter mappings (A through I)
    for i, char in enumerate("ABCDEFGHI"):
        data[0xC8 + i] = char

    # Special letter mappings (J, K, L)
    for i, char in enumerate("JKL"):
        data[0xF1 + i] = char

    return data


def parse(bff_str: str) -> bytearray:
    """Parse a BFF program string into a bytearray of instructions.

    The resulting program must be exactly 2 * SINGLE_TAPE_SIZE bytes long.
    """
    ret = bytearray()

    # Create mapping from op kinds to their byte values
    command_bytes = [0] * BffOp.NOOP.value
    for i in range(256):
        kind = get_op_kind(i)
        if kind.value < BffOp.NOOP.value:
            command_bytes[kind.value] = i

    i = 0
    while i < len(bff_str):
        if bff_str[i] == '0':
            ret.append(command_bytes[BffOp.NULL.value])
            i += 1
            continue

        found = False
        # Check if the character is a command
        for j in range(10):  # There are 10 command characters
            if bff_str[i] == command_repr()[j]:
                ret.append(command_bytes[j])
                i += 1
                found = True
                break

        if found:
            continue

        # Check if it's a character representation
        char_repr = character_repr()
        for j in range(256):
            s = char_repr[j]
            if i + len(s) <= len(bff_str) and bff_str[i:i+len(s)] == s:
                ret.append(j)
                i += len(s)
                found = True
                break

        if not found:
            print(f"Invalid BFF program, character {i} not recognized: {bff_str}")
            break

    # Enforce the tape size requirement
    if len(ret) != 2 * SINGLE_TAPE_SIZE:
        raise ValueError(f"Program must be exactly {2 * SINGLE_TAPE_SIZE} bytes long, but got {len(ret)} bytes")

    return ret


def map_char(c: int) -> str:
    """Map a byte to its string representation."""
    kind = get_op_kind(c)
    if kind.value < BffOp.NULL.value:  # Is command
        return command_repr()[kind.value]
    elif kind == BffOp.NULL:
        return "0"
    else:
        return character_repr()[c]


def get_foreground_color(kind: BffOp) -> str:
    """Get the appropriate foreground color based on operation kind."""
    if kind == BffOp.NULL:
        return Fore.RED
    elif kind.value < BffOp.NULL.value:  # Is command
        return Fore.LIGHTWHITE_EX  # Using bright white instead of regular white
    else:
        return Fore.LIGHTBLACK_EX


def get_background_color(i: int, head0_pos: int, head1_pos: int, pc_pos: int) -> str:
    """Get the appropriate background color based on position."""
    if i == pc_pos:
        return Back.GREEN
    elif i == head0_pos:
        return Back.BLUE
    elif i == head1_pos:
        return Back.RED
    else:
        return Back.BLACK


def print_program(head0_pos: int, head1_pos: int, pc_pos: int,
                  mem: bytearray,
                  separators: Optional[List[int]] = None) -> None:
    """Print the program with highlighting for the current positions."""
    if separators is None:
        separators = []

    sep_id = 0
    for i in range(len(mem)):
        if sep_id < len(separators) and separators[sep_id] == i:
            print("   ", end="")
            sep_id += 1

        c = mem[i]
        kind = get_op_kind(c)

        # Set colors based on character type and position
        fg_color = get_foreground_color(kind)
        bg_color = get_background_color(i, head0_pos, head1_pos, pc_pos)

        print(f"{fg_color}{bg_color}{map_char(c)}{Style.RESET_ALL}", end="")

    print()


def initial_state(tape: bytearray) -> Tuple[int, int, int]:
    """Set up the initial state with head positions and program counter.

    The initial state reads the first two bytes of the tape to determine
    the starting positions of the two heads, and sets the program counter
    to begin at position 2.
    """
    head0 = tape[0] % (2 * SINGLE_TAPE_SIZE)
    head1 = tape[1] % (2 * SINGLE_TAPE_SIZE)
    pc = 2
    return head0, head1, pc


def evaluate_one(tape: bytearray, head0: int, head1: int, pc: int) -> Tuple[int, int, int, bool]:
    """Evaluate a single instruction and update state.

    Returns:
        Tuple of (head0, head1, pc, is_command) where is_command indicates if
        the operation was a valid command or just a comment/noop.
    """
    cmd = tape[pc]
    kind = get_op_kind(cmd)

    if kind == BffOp.DEC0:
        head0 -= 1
    elif kind == BffOp.INC0:
        head0 += 1
    elif kind == BffOp.DEC1:
        head1 -= 1
    elif kind == BffOp.INC1:
        head1 += 1
    elif kind == BffOp.PLUS:
        tape[head0] = (tape[head0] + 1) & 0xFF  # Ensure 8-bit wrap
    elif kind == BffOp.MINUS:
        tape[head0] = (tape[head0] - 1) & 0xFF  # Ensure 8-bit wrap
    elif kind == BffOp.COPY01:
        tape[head1] = tape[head0]
    elif kind == BffOp.COPY10:
        tape[head0] = tape[head1]
    elif kind == BffOp.LOOP_START:
        if get_op_kind(tape[head0]) == BffOp.NULL:
            scanclosed = 1
            pc += 1
            while pc < (2 * SINGLE_TAPE_SIZE) and scanclosed > 0:
                if get_op_kind(tape[pc]) == BffOp.LOOP_END:
                    scanclosed -= 1
                if get_op_kind(tape[pc]) == BffOp.LOOP_START:
                    scanclosed += 1
                pc += 1
            pc -= 1
            if scanclosed != 0:
                pc = 2 * SINGLE_TAPE_SIZE  # Terminate execution
    elif kind == BffOp.LOOP_END:
        if get_op_kind(tape[head0]) != BffOp.NULL:
            scanopen = 1
            pc -= 1
            while pc >= 0 and scanopen > 0:
                if get_op_kind(tape[pc]) == BffOp.LOOP_END:
                    scanopen += 1
                if get_op_kind(tape[pc]) == BffOp.LOOP_START:
                    scanopen -= 1
                pc -= 1
            pc += 1
            if scanopen != 0:
                pc = -1  # Terminate execution
    else:
        return head0, head1, pc, False

    return head0, head1, pc, True


def evaluate(tape: bytearray, stepcount: int, debug: bool = False) -> int:
    """Evaluate the BFF program for a given number of steps.

    Args:
        tape: The program/memory tape
        stepcount: Maximum number of steps to execute
        debug: Whether to print debug information

    Returns:
        Number of actual steps executed (excluding noops)
    """
    # Verify tape size is exactly 2 * SINGLE_TAPE_SIZE
    if len(tape) != 2 * SINGLE_TAPE_SIZE:
        raise ValueError(f"Program must be exactly {2 * SINGLE_TAPE_SIZE} bytes long")

    nskip = 0

    head0_pos, head1_pos, pos = initial_state(tape)

    i = 0
    for i in range(stepcount):
        # Ensure head positions wrap around
        head0_pos = head0_pos & (2 * SINGLE_TAPE_SIZE - 1)
        head1_pos = head1_pos & (2 * SINGLE_TAPE_SIZE - 1)

        if debug:
            print_program(head0_pos, head1_pos, pos, tape)

        head0_pos, head1_pos, pos, is_command = evaluate_one(tape, head0_pos, head1_pos, pos)

        if not is_command:
            nskip += 1

        if pos < 0:
            i += 1
            break

        pos += 1
        if pos >= 2 * SINGLE_TAPE_SIZE:
            i += 1
            break

    return i - nskip

def evaluate_and_save(tape: bytearray, file_path: str, stepcount: int, debug: bool = False) -> int:
    """Evaluate the BFF program for a given number of steps and save each state to a file.

    The file format is a simple binary format:
    - 4-byte magic number ('BFF\0')
    - 4-byte format version (1)
    - 4-byte tape size
    - For each state:
      - 4-byte PC position
      - 4-byte head0 position
      - 4-byte head1 position
      - Tape contents (128 bytes)

    Args:
        tape: The program/memory tape
        file_path: Path to save the state history
        stepcount: Maximum number of steps to execute
        debug: Whether to print debug information

    Returns:
        Number of actual steps executed (excluding noops)
    """
    # Verify tape size is exactly 2 * SINGLE_TAPE_SIZE
    if len(tape) != 2 * SINGLE_TAPE_SIZE:
        raise ValueError(f"Program must be exactly {2 * SINGLE_TAPE_SIZE} bytes long")

    # Copy the initial tape to avoid modifying the original
    working_tape = bytearray(tape)

    nskip = 0
    head0_pos, head1_pos, pos = initial_state(working_tape)

    # Open the file for writing in binary mode
    with open(file_path, 'wb') as f:
        # Write header
        f.write(b'BFF\0')  # Magic number
        f.write(struct.pack('<I', 1))  # Format version
        f.write(struct.pack('<I', 2 * SINGLE_TAPE_SIZE))  # Tape size

        i = 0
        for i in range(stepcount):
            # Ensure head positions wrap around
            head0_pos = head0_pos & (2 * SINGLE_TAPE_SIZE - 1)
            head1_pos = head1_pos & (2 * SINGLE_TAPE_SIZE - 1)

            # Save the current state
            f.write(struct.pack('<I', pos))  # Program counter
            f.write(struct.pack('<I', head0_pos))  # Head 0 position
            f.write(struct.pack('<I', head1_pos))  # Head 1 position
            f.write(working_tape)  # Tape contents

            if debug:
                print_program(head0_pos, head1_pos, pos, working_tape)

            head0_pos, head1_pos, pos, is_command = evaluate_one(working_tape, head0_pos, head1_pos, pos)

            if not is_command:
                nskip += 1

            if pos < 0:
                i += 1
                break

            pos += 1
            if pos >= 2 * SINGLE_TAPE_SIZE:
                i += 1
                break

    return i - nskip

def read_and_display_states(file_path: str) -> None:
    """Read a BFF state file and display each state using print_program.

    Args:
        file_path: Path to the BFF state file
    """
    import struct

    with open(file_path, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != b'BFF\0':
            raise ValueError(f"Invalid file format: Expected 'BFF\\0', got {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        if version != 1:
            raise ValueError(f"Unsupported format version: {version}")

        tape_size = struct.unpack('<I', f.read(4))[0]
        if tape_size != 2 * SINGLE_TAPE_SIZE:
            raise ValueError(f"Unexpected tape size: {tape_size}, expected {2 * SINGLE_TAPE_SIZE}")

        # Read states
        state_count = 0
        while True:
            # Read program counter
            pc_bytes = f.read(4)
            if not pc_bytes or len(pc_bytes) < 4:
                break  # End of file

            pc = struct.unpack('<I', pc_bytes)[0]
            head0 = struct.unpack('<I', f.read(4))[0]
            head1 = struct.unpack('<I', f.read(4))[0]
            tape = bytearray(f.read(tape_size))

            if len(tape) < tape_size:
                print(f"Warning: Incomplete state at position {state_count}")
                break

            # Display the state
            print_program(head0, head1, pc, tape)
            state_count += 1

    print(f"\nTotal states read: {state_count}")


def test_evaluate_and_save(program: bytearray, steps: int = 100, max_display: int = 10) -> None:
    """Test the evaluate_and_save function by creating a temporary file, saving states, and displaying them.

    Args:
        program_str: BFF program string
        steps: Number of steps to execute
        max_display: Maximum number of states to display
    """
    import tempfile
    import os

    # Parse the program
    # program = parse(program_str)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bff') as temp_file:
        temp_path = temp_file.name

    try:
        # Run and save states
        print(f"Executing {steps} steps and saving to {temp_path}...")
        actual_steps = evaluate_and_save(program, temp_path, steps)
        print(f"Executed {actual_steps} actual steps (excluding noops)")

        # Read and display states
        print("\nReading saved states:")
        read_and_display_states(temp_path, max_display)

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"\nRemoved temporary file: {temp_path}")
