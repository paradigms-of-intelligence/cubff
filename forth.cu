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
}  // namespace
