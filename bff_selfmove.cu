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

#define BFF_CUSTOM_LOGIC
#define BFF_CUSTOM_OPS
#define BFF_HEADS
#include "bff.inc.h"

namespace {

const char *Bff::name() { return "bff_selfmove"; }

__device__ __host__ BffOp Bff::GetOpKind(char c) {
  switch (c) {
    case 0:
      return BffOp::kInc0;
    case 1:
      return BffOp::kDec0;
    case 2:
      return BffOp::kPlus;
    case 3:
      return BffOp::kMinus;
    case 4:
      return BffOp::kCopy10;
    case 5:
      return BffOp::kLoopStart;
    case 6:
      return BffOp::kLoopEnd;
    default:
      return BffOp::kNoop;
  }
}

bool __device__ Bff::EvaluateOne(uint8_t *tape, int &head0, int &head1,
                                 int &pc) {
  char cmd = tape[pc];
  switch (GetOpKind(cmd)) {
    case BffOp::kDec0:
      head0--;
      break;
    case BffOp::kInc0:
      head0++;
      break;
      break;
    case BffOp::kPlus:
      tape[head0]++;
      break;
    case BffOp::kMinus:
      tape[head0]--;
      break;
    case BffOp::kCopy10:
      tape[head0] = tape[head1];
      head1++;
      break;
    case BffOp::kLoopStart:
      if (!tape[head0]) {
        size_t scanclosed = 1;
        pc++;
        for (; pc < (2 * kSingleTapeSize) && scanclosed > 0; pc++) {
          if (GetOpKind(tape[pc]) == BffOp::kLoopEnd) scanclosed--;
          if (GetOpKind(tape[pc]) == BffOp::kLoopStart) scanclosed++;
        }
        pc--;
        if (scanclosed != 0) {
          pc = 2 * kSingleTapeSize;
        }
      }
      break;
    case BffOp::kLoopEnd:
      if (tape[head0]) {
        size_t scanopen = 1;
        pc--;
        for (; pc >= 0 && scanopen > 0; pc--) {
          if (GetOpKind(tape[pc]) == BffOp::kLoopEnd) scanopen++;
          if (GetOpKind(tape[pc]) == BffOp::kLoopStart) scanopen--;
        }
        pc++;
        if (scanopen != 0) {
          pc = -1;
        }
      }
      break;
    default:
      return false;
  }
  return true;
}
}  // namespace
