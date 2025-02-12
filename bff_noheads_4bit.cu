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

#define BFF_CUSTOM_OPS
#include "bff.inc.h"

namespace {

const char *Bff::name() { return "bff_noheads_4bit"; }

__device__ __host__ BffOp Bff::GetOpKind(char c) {
  switch (((int)c + 256) % 16) {
    case 6:
      return BffOp::kLoopStart;
    case 7:
      return BffOp::kLoopEnd;
    case 8:
      return BffOp::kPlus;
    case 9:
      return BffOp::kMinus;
    case 10:
      return BffOp::kCopy01;
    case 11:
      return BffOp::kCopy10;
    case 12:
      return BffOp::kDec0;
    case 13:
      return BffOp::kInc0;
    case 14:
      return BffOp::kDec1;
    case 15:
      return BffOp::kInc1;
    case 0:
      return BffOp::kNull;
    default:
      return BffOp::kNoop;
  }
}

}  // namespace
