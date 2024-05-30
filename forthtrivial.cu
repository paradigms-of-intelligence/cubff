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
const char *Forth::name() { return "forthtrivial"; }

void Forth::InitByteColors(
    std::array<std::array<uint8_t, 3>, 256> &byte_colors) {
  // I/O
  for (auto i : {0x00, 0x01, 0x02, 0x03, 0x0C, 0x0D}) {
    byte_colors[i] = {200, 0, 200};
  }
  // Stack manipulation
  for (auto i : {0x04, 0x05, 0x06, 0x08, 0x09, 0x0A, 0x0B}) {
    byte_colors[i] = {0, 128, 200};
  }
  // Conditional jump
  byte_colors[0x07] = {255, 0, 0};
  // Forward jump
  for (size_t i = 0b10'000000; i < 0b11'000000; i++) {
    uint8_t v = 128 + (i - 0b10'000000) / 2;
    byte_colors[i] = {0, v, v};
  }
  // Backward jump
  for (size_t i = 0b11'000000; i < 0b100'000000; i++) {
    uint8_t v = 128 + (i - 0b11'000000) / 2;
    byte_colors[i] = {0, 0, v};
  }
  // Constant
  for (size_t i = 0b01'000000; i < 0b10'000000; i++) {
    uint8_t v = 192 + (i - 0b01'000000) / 2;
    byte_colors[i] = {v, v, v};
  }
  // Comment
  for (size_t i = 0x0E; i < 0x40; i++) {
    uint8_t v = (i - 0x0E) / 2;
    byte_colors[i] = {v, v, v};
  }
}

__device__ void Forth::EvaluateOne(uint8_t *tape, int &pos, size_t &nops,
                                   Stack &stack) {
  // 000000xy (00-03) -> (read|write)(0|1)
  // 00000100 (04)    -> dup
  // 00000101 (05)    -> drop
  // 00000110 (06)    -> swap
  // 00000111 (07)    -> if0
  // 00001000 (08)    -> inc
  // 00001001 (09)    -> dec
  // 00001010 (0A)    -> add
  // 00001011 (0B)    -> sub
  // 0000110x (0C-0D) -> copy(0->1)(1->0)
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
      case 0x0:
      case 0x1: {
        int t = command & 1;
        int addr = stack.Pop() % kSingleTapeSize;
        stack.Push(tape[(t ? kSingleTapeSize : 0) + addr]);
        break;
      }
      case 0x2:
      case 0x3: {
        int t = command & 1;
        int val = stack.Pop();
        int addr = stack.Pop() % kSingleTapeSize;
        tape[(t ? kSingleTapeSize : 0) + addr] = val;
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
      case 0xC:
      case 0xD: {
        int dir = command & 1;
        int addr = stack.Pop() % kSingleTapeSize;
        tape[(dir ? 0 : kSingleTapeSize) + addr] =
            tape[(dir ? kSingleTapeSize : 0) + addr];
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
