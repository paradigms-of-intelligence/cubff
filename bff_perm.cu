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

#include "common_language.h"

namespace {

// define the instruction set of bff in the order <>{}+-.,[]
#define INST0 1
#define INST1 2
#define INST2 3
#define INST3 4
#define INST4 5
#define INST5 6
#define INST6 7
#define INST7 8
#define INST8 9
#define INST9 10

__device__ __host__ uint32_t headpos(uint32_t b) {
  return b % (2 * kSingleTapeSize);
}

__host__ __device__ bool isin(const char *chars, char c) {
  for (; *chars; chars++) {
    if (c == *chars) return true;
  }
  return false;
}

struct BffPerm {
  static const char *name() { return "bff_perm"; }

  static void InitByteColors(
      std::array<std::array<uint8_t, 3>, 256> &byte_colors) {
    for (size_t i = 0; i < 256; i++) {
      byte_colors[i][0] = 255;
      byte_colors[i][1] = 255;
      byte_colors[i][2] = 255;
    }
  }

  static std::string Parse(std::string bff) { assert(false); }

  static __device__ __host__ const char *MapChar(char c, char *chmem) {
    constexpr char print_inst[10] = {'<', '>', '{', '}', '+',
                                     '-', '.', ',', '[', ']'};
    chmem[1] = 0;
    chmem[0] = c;
    if (chmem[0] == 0) {
      return "␀";
    }
    switch (chmem[0]) {
      case INST0:
        chmem[0] = print_inst[0];
        break;
      case INST1:
        chmem[0] = print_inst[1];
        break;
      case INST2:
        chmem[0] = print_inst[2];
        break;
      case INST3:
        chmem[0] = print_inst[3];
        break;
      case INST4:
        chmem[0] = print_inst[4];
        break;
      case INST5:
        chmem[0] = print_inst[5];
        break;
      case INST6:
        chmem[0] = print_inst[6];
        break;
      case INST7:
        chmem[0] = print_inst[7];
        break;
      case INST8:
        chmem[0] = print_inst[8];
        break;
      case INST9:
        chmem[0] = print_inst[9];
        break;
      default:
        chmem[0] = ' ';
    }
    return chmem;
  }

  static __device__ __host__ void PrintProgramInternal(
      size_t head0_pos, size_t head1_pos, size_t pc_pos, const uint8_t *mem,
      size_t len, const uint8_t *mem2, size_t len2) {
    auto print_char = [&](char c, size_t i) {
      char chmem[32] = {};
      const char *cc = MapChar(mem[i], chmem);
      bool is_command = isin("<>{}+-.,[]", *cc);
      if (i == head0_pos) {
        printf("\x1b[44;1m");
      }
      if (i == head1_pos) {
        printf("\x1b[41;1m");
      }
      if (i == pc_pos) {
        printf("\x1b[42;1m");
      }
      if (is_command) {
        printf("\x1b[37;1m");
      }
      printf("%s", cc);
      if (is_command || i == head0_pos || i == head1_pos || i == pc_pos) {
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
    size_t head0_pos = headpos(mem[0]);
    size_t head1_pos = headpos(mem[1]);
    PrintProgramInternal(head0_pos, head1_pos, pc_pos, mem, len, mem2, len2);
  }

  static __device__ size_t Evaluate(uint8_t *tape, size_t stepcount,
                                    bool debug) {
    size_t nskip = 0;

    int pos = 2;
    int head0_pos = headpos(tape[0]);
    int head1_pos = headpos(tape[1]);

    size_t i = 0;

    for (; i < stepcount; i++) {
      head0_pos = head0_pos & (2 * kSingleTapeSize - 1);
      head1_pos = head1_pos & (2 * kSingleTapeSize - 1);
      if (debug) {
        PrintProgramInternal(head0_pos, head1_pos, pos, tape,
                             2 * kSingleTapeSize, nullptr, 0);
      }
      uint8_t cmd = tape[pos];
      switch (cmd) {
        case INST0:
          head0_pos--;
          break;
        case INST1:
          head0_pos++;
          break;
        case INST2:
          head1_pos--;
          break;
        case INST3:
          head1_pos++;
          break;
        case INST4:
          tape[head0_pos]++;
          break;
        case INST5:
          tape[head0_pos]--;
          break;
        case INST6:
          tape[head1_pos] = tape[head0_pos];
          break;
        case INST7:
          tape[head0_pos] = tape[head1_pos];
          break;
        case INST8:
          if (!tape[head0_pos]) {
            size_t scanclosed = 1;
            pos++;
            for (; pos < (2 * kSingleTapeSize) && scanclosed > 0; pos++) {
              if (tape[pos] == INST9) scanclosed--;
              if (tape[pos] == INST8) scanclosed++;
            }
            pos--;
            if (scanclosed != 0) {
              pos = 2 * kSingleTapeSize;
            }
          }
          break;
        case INST9:
          if (tape[head0_pos]) {
            size_t scanopen = 1;
            pos--;
            for (; pos >= 0 && scanopen > 0; pos--) {
              if (tape[pos] == INST9) scanopen++;
              if (tape[pos] == INST8) scanopen--;
            }
            pos++;
            if (scanopen != 0) {
              pos = -1;
            }
          }
          break;
        default:
          nskip++;
      }
      if (pos < 0) {
        i++;
        break;
      }
      pos++;
      if (pos >= 2 * kSingleTapeSize) {
        i++;
        break;
      }
    }

    return i - nskip;
  }
};

REGISTER(Bff_perm);
}  // namespace
