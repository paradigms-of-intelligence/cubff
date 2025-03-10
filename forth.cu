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
const char *Forth::name() { return "forth"; }

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
  // 000000xy (00-03) -> (read|write)(0|1)
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
  switch (command) {
    case kRead0:
    case kRead1: {
      int t = command & 1;
      int addr = stack.Pop() % kSingleTapeSize;
      stack.Push(tape[(t ? kSingleTapeSize : 0) + addr]);
      break;
    }
    case kWrite0:
    case kWrite1: {
      int t = command & 1;
      int val = stack.Pop();
      int addr = stack.Pop() % kSingleTapeSize;
      tape[(t ? kSingleTapeSize : 0) + addr] = val;
      break;
    }
    case kDup: {
      int v = stack.Pop();
      stack.Push(v);
      stack.Push(v);
      break;
    }
    case kDrop:
      stack.Pop();
      break;
    case kSwap: {
      int a = stack.Pop();
      int b = stack.Pop();
      stack.Push(a);
      stack.Push(b);
      break;
    }
    case kIf0: {
      int v = stack.Pop();
      if (v) {
        pos++;
      }
      stack.Push(v);
      break;
    }
    case kInc: {
      stack.Push(stack.Pop() + 1);
      break;
    }
    case kDec: {
      stack.Push(stack.Pop() - 1);
      break;
    }
    case kAdd: {
      int a = stack.Pop();
      int b = stack.Pop();
      stack.Push(a + b);
      break;
    }
    case kSub: {
      int a = stack.Pop();
      int b = stack.Pop();
      stack.Push(a - b);
      break;
    }
    case kConst: {
      stack.Push(command & 63);
      pos++;
      break;
    }
    case kJmp: {
      int abs = (command & 63) + 1;
      int jmp = command & 64 ? -abs : abs;
      pos += jmp;
      pos--;
      break;
    }

    default: {
      nops++;
    }
  }
  pos++;
}
}  // namespace
