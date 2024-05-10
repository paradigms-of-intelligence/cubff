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

#include "common_language.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

struct Forth {
  static const char *name();

  static void
  InitByteColors(std::array<std::array<uint8_t, 3>, 256> &byte_colors);

  static std::string Parse(std::string hex) {
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

  static __device__ __host__ void
  PrintProgramInternal(size_t pc_pos, const uint8_t *mem, size_t len,
                       const uint8_t *mem2, size_t len2, const uint8_t *stack,
                       size_t stack_len) {
    auto print_char = [&](char c, size_t i) {
      bool color = false;
      if (i == pc_pos) {
        printf("\x1b[33;1m");
        color = true;
      }
      char chmem[4];
      printf("%s", MapChar(c, chmem));
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

  static void PrintProgram(size_t pc_pos, const uint8_t *mem, size_t len,
                           const uint8_t *mem2, size_t len2) {
    PrintProgramInternal(pc_pos, mem, len, mem2, len2, nullptr, 0);
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
} // namespace
