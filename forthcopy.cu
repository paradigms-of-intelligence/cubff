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

#define FORTH_CUSTOM_OPS
#include "forth.inc.h"

namespace {
const char *Forth::name() { return "forthcopy"; }

__device__ __host__ ForthOp Forth::GetOpKind(uint8_t c) {
  switch (c) {
    case 0x0:
      return ForthOp::kRead;
    case 0x1:
      return ForthOp::kWrite;
    case 0x2:
      return ForthOp::kCopy;
    case 0x3:
      return ForthOp::kXor;
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
    default:
      return (c >= 128 ? ForthOp::kJmp
                       : (c >= 64 ? ForthOp::kConst : ForthOp::kNoop));
  }
}

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

}  // namespace
