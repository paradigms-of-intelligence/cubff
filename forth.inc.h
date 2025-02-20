/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "common_language.h"

namespace {

struct Forth {
  static const char *name();

  static void InitByteColors(
      std::array<std::array<uint8_t, 3>, 256> &byte_colors);

  static std::vector<uint8_t> Parse(std::string hex) {
    std::vector<uint8_t> ret;

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

    for (size_t i = 0; i < hex.size(); i += 2) {
      ret.push_back(fromhex(hex[i]) * 16 + fromhex(hex[i + 1]));
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
      size_t pc_pos, const uint8_t *mem, size_t len, const size_t *separators,
      size_t num_separators, const uint8_t *stack, size_t stack_len) {
    auto print_char = [&](char c, size_t i) {
      if (i == pc_pos) {
        printf("\x1b[48:5:22m");
      }
      char chmem[4];
      printf("\x1b[38:5:250m%s%s", MapChar(c, chmem), ResetColors());
    };
    size_t sep_id = 0;
    for (size_t i = 0; i < len; i++) {
      if (sep_id < num_separators && separators[sep_id] == i) {
        printf("   ");
        sep_id++;
      }
      char c = mem[i];
      print_char(c, i);
    }
    if (stack) {
      printf("   ");
      for (size_t i = 0; i < stack_len; i++) {
        char c = stack[i];
        char chmem[4];
        printf("%s ", MapChar(c, chmem));
      }
    }
    printf("\n");
  }

  static __device__ __host__ void PrintProgram(size_t pc_pos,
                                               const uint8_t *mem, size_t len,
                                               const size_t *separators,
                                               size_t num_separators) {
    PrintProgramInternal(pc_pos, mem, len, separators, num_separators, nullptr,
                         0);
  }

  struct Stack {
    static constexpr size_t kStackSize = 128;
    uint8_t data[kStackSize] = {};
    size_t stackpos = 0;
    bool overflow = false;
    __device__ void Push(uint8_t val) {
      if (stackpos != kStackSize) {
        data[stackpos++] = val;
      } else {
        overflow = true;
      }
    };
    __device__ uint8_t Pop() { return stackpos == 0 ? 0 : data[--stackpos]; };
  };

  static __device__ void EvaluateOne(uint8_t *tape, int &pos, size_t &nops,
                                     Stack &stack);
  static __device__ size_t Evaluate(uint8_t *tape, size_t stepcount,
                                    bool debug) {
    Stack stack;
    int pos = 0;
    size_t i = 0;
    size_t nops = 0;
    for (; i < stepcount; i++) {
      if (debug) {
        PrintProgramInternal(pos, tape, 2 * kSingleTapeSize, nullptr, 0,
                             stack.data, stack.stackpos);
      }
      EvaluateOne(tape, pos, nops, stack);
      if (pos >= 2 * kSingleTapeSize || pos < 0 || stack.overflow) {
        i++;
        break;
      }
    }
    return i - nops;
  }
};

REGISTER(Forth);
}  // namespace
