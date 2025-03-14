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

#include <cstddef>
#include <cstdint>
#define FORTH_CUSTOM_OPS
#define FORTH_CUSTOM_LOGIC
#include <array>

#include "forth.inc.h"
__device__ __host__ ForthOp Forth::GetOpKind(uint8_t c) {
  switch (c) {
    case 0x0:
      return ForthOp::kRead0;
    case 0x1:
      return ForthOp::kRead1;
    case 0x2:
      return ForthOp::kWrite0;
    case 0x3:
      return ForthOp::kWrite1;
    case 0x4:
      return ForthOp::kDup;
    case 0x5:
      return ForthOp::kDrop;
    case 0x6:
      return ForthOp::kSwap;
    case 0x7:
      return ForthOp::kIf0;
    case 0x8:
      return ForthOp::kInc;
    case 0x9:
      return ForthOp::kDec;
    case 0xA:
      return ForthOp::kAdd;
    case 0xB:
      return ForthOp::kSub;
    case 0xC:
      return ForthOp::kCopy0;
    case 0xD:
      return ForthOp::kCopy1;
    default:
      return (c >= 128 ? ForthOp::kJmp
                       : (c >= 64 ? ForthOp::kConst : ForthOp::kNoop));
  }
}

__device__ void Reset(uint8_t *tape, int pos) {
  uint64_t seed = 0;
  for (size_t i = 0; i < 2 * kSingleTapeSize; i++) {
    seed = SplitMix64(seed ^ tape[i]);
  }
  for (size_t i = 0; i < kSingleTapeSize; i++) {
    tape[i] = SplitMix64((seed)*kSingleTapeSize * kSingleTapeSize +
                         pos * kSingleTapeSize + i) %
              256;
  }
}

__device__ bool ResetIf(uint8_t *tape, int addr) {
  return tape[kSingleTapeSize + addr] == 0x13;
}

namespace {

const char *Forth::name() { return "forthtrivial_reset"; }

void Forth::InitByteColors(
    std::array<std::array<uint8_t, 3>, 256> &byte_colors) {
  auto scale_color = [](std::array<uint8_t, 3> &color, size_t offset,
                        size_t num) {
    float darken_amount = offset * 0.2f / (num - 1);
    float multiplier = 1.0f - darken_amount;
    for (size_t c = 0; c < 3; c++) {
      color[c] =
          std::round(std::min(255.0f, std::max(0.0f, multiplier * color[c])));
    }
  };

  // I/O
  for (auto i : {0x00, 0x01, 0x02, 0x03, 0x0C, 0x0D}) {
    byte_colors[i] = {0x73, 0x01, 0xce};
  }
  // Stack manipulation
  for (auto i : {0x04, 0x05, 0x06, 0x08, 0x09, 0x0A, 0x0B}) {
    byte_colors[i] = {0x4e, 0x10, 0x01};
  }
  // Conditional jump
  byte_colors[0x07] = {0x00, 0x00, 0x00};
  // Forward jump
  for (size_t i = 0b10'000000; i < 0b11'000000; i++) {
    byte_colors[i] = {0x94, 0xd9, 0xff};
    scale_color(byte_colors[i], i - 0b10'000000, 0b1'000000);
  }
  // Backward jump
  for (size_t i = 0b11'000000; i < 0b100'000000; i++) {
    byte_colors[i] = {0xff, 0x77, 0x7d};
    scale_color(byte_colors[i], i - 0b11'000000, 0b1'000000);
  }
  // Constant
  for (size_t i = 0b01'000000; i < 0b10'000000; i++) {
    byte_colors[i] = {0xff, 0xff, 0xff};
    scale_color(byte_colors[i], i - 0b1'000000, 0b1'000000);
  }
  // Comment
  for (size_t i = 0x0E; i < 0x40; i++) {
    byte_colors[i] = {0x02, 0x8a, 0x37};
    scale_color(byte_colors[i], i - 0x0E, 50);
  }
}

__device__ void Forth::EvaluateOne(uint8_t *tape, int &pos, size_t &nops,
                                   Forth::Stack &stack) {
  uint8_t command = tape[pos];
  switch (Forth::GetOpKind(command)) {
    case kRead0: {
      int addr = stack.Pop() % kSingleTapeSize;
      stack.Push(tape[0 + addr]);
      break;
    }
    case kRead1: {
      int addr = stack.Pop() % kSingleTapeSize;
      stack.Push(tape[kSingleTapeSize + addr]);
      break;
    }
    case kRead: {
      int addr = stack.Pop() % (2 * kSingleTapeSize);
      stack.Push(tape[addr]);
      break;
    }
    case kWrite: {
      int val = stack.Pop();
      int addr = stack.Pop() % (2 * kSingleTapeSize);
      tape[addr] = val;
      break;
    }

    case kWrite0: {
      int val = stack.Pop();
      int addr = stack.Pop() % kSingleTapeSize;
      tape[0 + addr] = val;
      break;
    }
    case kWrite1: {
      int val = stack.Pop();
      int addr = stack.Pop() % kSingleTapeSize;
      if (ResetIf(tape, addr)) {
        Reset(tape, pos);
      } else {
        tape[kSingleTapeSize + addr] = val;
      }
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
      // 01xxxxxx (40-7F) -> stack.Push unsigned constant xxxxxx
      stack.Push(command & 63);
      break;
    }
    case kCopy0: {
      int addr = stack.Pop() % kSingleTapeSize;
      if (ResetIf(tape, addr)) {
        Reset(tape, pos);
      } else {
        tape[kSingleTapeSize + addr] = tape[0 + addr];
      }
      break;
    }
    case kCopy1: {
      int addr = stack.Pop() % kSingleTapeSize;
      tape[0 + addr] = tape[kSingleTapeSize + addr];
      break;
    }
    case kCopy: {
      int to = stack.Pop() % (2 * kSingleTapeSize);
      int from = stack.Pop() % (2 * kSingleTapeSize);
      tape[to] = tape[from];
      break;
    }
    case kXor: {
      stack.Push(stack.Pop() ^ 64);
      break;
    }

    case kJmp: {
      // 1Xxxxxxx (80-FF) -> jump to offset {+-}(xxxxxx+1)
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
