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
#define BFF_HEADS
#include "bff.inc.h"

namespace {

const char *Bff::name() { return "bff_perm"; }

__device__ __host__ BffOp Bff::GetOpKind(char c) {
  switch (c) {
    case 0:
      return BffOp::kNull;
    case 9:
      return BffOp::kLoopStart;
    case 10:
      return BffOp::kLoopEnd;
    case 5:
      return BffOp::kPlus;
    case 6:
      return BffOp::kMinus;
    case 7:
      return BffOp::kCopy01;
    case 8:
      return BffOp::kCopy10;
    case 1:
      return BffOp::kDec0;
    case 2:
      return BffOp::kInc0;
    case 3:
      return BffOp::kDec1;
    case 4:
      return BffOp::kInc1;
    default:
      return BffOp::kNoop;
  }
}
}  // namespace
