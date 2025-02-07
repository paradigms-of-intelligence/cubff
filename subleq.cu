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
#include <cstdio>

#include "common_language.h"

namespace {

struct Subleq {
  static const char *name() { return "subleq"; }

  static void InitByteColors(
      std::array<std::array<uint8_t, 3>, 256> &byte_colors) {
    for (size_t i = 0; i < 256; i++) {
      byte_colors[i][0] = i;
      byte_colors[i][1] = 0;
      byte_colors[i][2] = 255 - i;
    }
  }

  static std::string Parse(std::string subleq) {
    std::string ret;

    auto fromhex = [](char c) {
      if ('0' <= c && c <= '9') {
        return c - '0';
      }
      if ('a' <= c && c <= 'f') {
        return c - 'a' + 10;
      }
      if ('A' <= c && c <= 'F') {
        return c - 'A' + 10;
      }
      fprintf(stderr, "Unknown input character: %c\n", c);
      exit(1);
    };

    for (size_t i = 0; i < subleq.size(); i += 2) {
      ret.push_back(fromhex(subleq[i]) * 16 + fromhex(subleq[i + 1]));
    }
    return ret;
  }

  static __device__ __host__ const char *MapChar(char c, char *chmem) {
    unsigned x = (unsigned char)c;
    static const char *chars = "0123456789ABCDEF";
    chmem[0] = chars[x / 16];
    chmem[1] = chars[x % 16];
    chmem[2] = 0;
    return chmem;
  }

  static __device__ __host__ void PrintProgramInternal(
      size_t head0_pos, size_t head1_pos, size_t pc_pos, const uint8_t *mem,
      size_t len, const uint8_t *mem2, size_t len2) {
    auto print_char = [&](char c, size_t i) {
      bool color = false;
      if (i == pc_pos || i == pc_pos + 1 || i == pc_pos + 2) {
        printf("\x1b[32;1m");
        color = true;
      }
      if (i == head0_pos) {
        printf("\x1b[34;1m");
        color = true;
      }
      if (i == head1_pos) {
        printf("\x1b[31;1m");
        color = true;
      }
      printf("%02X", c < 0 ? (int)c + 256 : c);
      if (color) {
        printf("\x1b[;m");
      }
    };
    for (size_t i = 0; i < len; i++) {
      char c = mem[i];
      print_char(c, i);
    }
    if (mem2) {
      printf("   ");
      for (size_t i = len; i < len + len2; i++) {
        char c = mem2[i - len];
        print_char(c, i);
      }
    }
    printf("\n");
  }

  static void PrintProgram(size_t pc_pos, const uint8_t *mem, size_t len,
                           const uint8_t *mem2, size_t len2) {
    PrintProgramInternal(2 * kSingleTapeSize, 2 * kSingleTapeSize, pc_pos, mem,
                         len, mem2, len2);
  }

  static __device__ size_t Evaluate(uint8_t *tape, size_t stepcount,
                                    bool debug) {
    int pos = 0;
    size_t i = 0;
    for (; i < stepcount; i++) {
      size_t sa = tape[pos] % (2 * kSingleTapeSize);
      size_t sb = tape[pos + 1] % (2 * kSingleTapeSize);
      if (debug) {
        PrintProgramInternal(sa, sb, pos, tape, 2 * kSingleTapeSize, nullptr,
                             0);
      }
      tape[sa] -= tape[sb];
      if (tape[sa] & 0x80 || tape[sa] == 0) {
        pos = tape[pos + 2];
      } else {
        pos += 3;
      }
      if (pos + 3 > 2 * kSingleTapeSize || pos < 0) {
        i++;
        break;
      }
    }

    return i;
  }
};

REGISTER(Subleq);
}  // namespace
