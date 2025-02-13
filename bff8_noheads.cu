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
#include "bff.inc.h"

namespace {

const char *Bff::name() { return "bff8_noheads"; }

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
    case BffOp::kDec1:
      head1--;
      break;
    case BffOp::kInc1:
      head1++;
      break;
    case BffOp::kPlus:
      tape[head0]++;
      break;
    case BffOp::kMinus:
      tape[head0]--;
      break;
    case BffOp::kCopy01:
      tape[head1] = tape[head0];
      break;
    case BffOp::kCopy10:
      tape[head0] = tape[head1];
      break;
    case BffOp::kLoopStart:
      if (!tape[head0]) {
        pc += 8;
      }
      break;
    case BffOp::kLoopEnd:
      if (tape[head0]) {
        pc -= 8;
      }
      break;
    default:
      return false;
  }
  return true;
}

}  // namespace
