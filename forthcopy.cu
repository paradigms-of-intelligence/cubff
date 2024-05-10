// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "forth.inc.h"

namespace {
const char *Forth::name() { return "forthcopy"; }

void Forth::InitByteColors(
    std::array<std::array<uint8_t, 3>, 256> &byte_colors) {
  for (size_t i = 0; i < 256; i++) {
    byte_colors[i][0] = i;
    byte_colors[i][1] = 0;
    byte_colors[i][2] = 255 - i;
  }
  for (size_t i = 0; i < 0xC; i++) {
    byte_colors[i][1] = 128;
  }
}

__device__ void Forth::EvaluateOne(uint8_t *tape, int &pos, size_t &nops,
                                   Stack &stack) {
  // 00000000 (00)    -> read
  // 00000001 (01)    -> write
  // 00000010 (02)    -> copy
  // 00000011 (03)    -> ^ 64
  // 00000100 (04)    -> dup
  // 00000101 (05)    -> drop
  // 00000110 (06)    -> swap
  // 00000111 (07)    -> if0
  // 00001000 (08)    -> inc
  // 00001001 (09)    -> dec
  // 00001010 (0A)    -> add
  // 00001011 (0B)    -> sub
  // 01xxxxxx (40-7F) -> stack.Push unsigned constant xxxxxx
  // 1Xxxxxxx (80-FF) -> jump to offset {+-}(xxxxxx+1)
  uint8_t command = tape[pos];
  if (command >= 128) {
    int abs = (command & 63) + 1;
    int jmp = command & 64 ? -abs : abs;
    pos += jmp;
  } else if (command >= 64) {
    stack.Push(command & 63);
    pos++;
  } else {
    switch (command) {
      case 0x0: {
        int addr = stack.Pop() % (2 * kSingleTapeSize);
        stack.Push(tape[addr]);
        break;
      }
      case 0x1: {
        int val = stack.Pop();
        int addr = stack.Pop() % (2 * kSingleTapeSize);
        tape[addr] = val;
        break;
      }
      case 0x2: {
        int to = stack.Pop() % (2 * kSingleTapeSize);
        int from = stack.Pop() % (2 * kSingleTapeSize);
        tape[to] = tape[from];
        break;
      }
      case 0x3: {
        stack.Push(stack.Pop() ^ 64);
        break;
      }
      case 0x4: {
        int v = stack.Pop();
        stack.Push(v);
        stack.Push(v);
        break;
      }
      case 0x5:
        stack.Pop();
        break;
      case 0x6: {
        int a = stack.Pop();
        int b = stack.Pop();
        stack.Push(a);
        stack.Push(b);
        break;
      }
      case 0x7: {
        int v = stack.Pop();
        if (v) {
          pos++;
        }
        stack.Push(v);
        break;
      }
      case 0x8: {
        stack.Push(stack.Pop() + 1);
        break;
      }
      case 0x9: {
        stack.Push(stack.Pop() - 1);
        break;
      }
      case 0xA: {
        int a = stack.Pop();
        int b = stack.Pop();
        stack.Push(a + b);
        break;
      }
      case 0xB: {
        int a = stack.Pop();
        int b = stack.Pop();
        stack.Push(a - b);
        break;
      }
      default: {
        nops++;
      }
    }
    pos++;
  }
}
}  // namespace
