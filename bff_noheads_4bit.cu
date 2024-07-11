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

__host__ __device__ bool isin(const char *chars, char c) {
  for (; *chars; chars++) {
    if (c == *chars) return true;
  }
  return false;
}

struct Bff {
  static const char *name() { return "bff_noheads_4bit"; }

  static void InitByteColors(
      std::array<std::array<uint8_t, 3>, 256> &byte_colors) {
    for (size_t i = 0; i < 16; i++) {
      byte_colors[i * 16 + 0] = {255, 0, 0};
      for (size_t j = 1; j < 6; j++) {
        byte_colors[i * 16 + j] = {192, 192, 192};
      }
      byte_colors[i * 16 + 6] = byte_colors[i * 16 + 7] = {0, 192, 0};
      byte_colors[i * 16 + 8] = byte_colors[i * 16 + 9] = {200, 0, 200};
      byte_colors[i * 16 + 10] = byte_colors[i * 16 + 11] = {200, 0, 200};
      byte_colors[i * 16 + 12] = byte_colors[i * 16 + 13] = {0, 128, 220};
      byte_colors[i * 16 + 14] = byte_colors[i * 16 + 15] = {0, 128, 220};
    }
  }

  static std::string Parse(std::string bff) {
    std::string ret;
    char reverse_map[256] = {};
    for (size_t i = 0; i < 16; i++) {
      char chmem[32];
      reverse_map[(int)MapChar(i, chmem)[0]] = i;
    }
    for (size_t i = 0; i < bff.size();) {
      ret.push_back(reverse_map[(int)bff[i]]);
      i++;
    }
    return ret;
  }

  static __device__ __host__ const char *MapChar(char c, char *chmem) {
    constexpr char kCharMap[16] = {'0', '1', '2', '3', '4', '5', '[', ']',
                                   '+', '-', '.', ',', '<', '>', '{', '}'};
    chmem[1] = 0;
    chmem[0] = kCharMap[((int)c + 256) % 16];
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
    size_t head0_pos = 2 * kSingleTapeSize;
    size_t head1_pos = 2 * kSingleTapeSize;
    PrintProgramInternal(head0_pos, head1_pos, pc_pos, mem, len, mem2, len2);
  }

  static __device__ size_t Evaluate(uint8_t *tape, size_t stepcount,
                                    bool debug) {
    size_t nskip = 0;

    int pos = 0;
    int head0_pos = 0;
    int head1_pos = 0;

    size_t i = 0;
    for (; i < stepcount; i++) {
      head0_pos = head0_pos & (2 * kSingleTapeSize - 1);
      head1_pos = head1_pos & (2 * kSingleTapeSize - 1);
      if (debug) {
        PrintProgramInternal(head0_pos, head1_pos, pos, tape,
                             2 * kSingleTapeSize, nullptr, 0);
      }
      auto t = [&](size_t pos) { return tape[pos] % 16; };
      char cmd = t(pos);
      switch (cmd) {
        case 12:
          head0_pos--;
          break;
        case 13:
          head0_pos++;
          break;
        case 14:
          head1_pos--;
          break;
        case 15:
          head1_pos++;
          break;
        case 8:
          tape[head0_pos]++;
          break;
        case 9:
          tape[head0_pos]--;
          break;
        case 10:
          tape[head1_pos] = tape[head0_pos];
          break;
        case 11:
          tape[head0_pos] = tape[head1_pos];
          break;
        case 6:
          if (!t(head0_pos)) {
            size_t scanclosed = 1;
            pos++;
            for (; pos < (2 * kSingleTapeSize) && scanclosed > 0; pos++) {
              if (t(pos) == 7) scanclosed--;
              if (t(pos) == 6) scanclosed++;
            }
            pos--;
            if (scanclosed != 0) {
              pos = 2 * kSingleTapeSize;
            }
          }
          break;
        case 7:
          if (t(head0_pos)) {
            size_t scanopen = 1;
            pos--;
            for (; pos >= 0 && scanopen > 0; pos--) {
              if (t(pos) == 7) scanopen++;
              if (t(pos) == 6) scanopen--;
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

REGISTER(Bff);
}  // namespace
