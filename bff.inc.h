/*
 * Copyright 2025 Google LLC
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

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "common_language.h"

namespace {

enum BffOp {
  kLoopStart,
  kLoopEnd,
  kPlus,
  kMinus,
  kCopy01,
  kCopy10,
  kDec0,
  kInc0,
  kDec1,
  kInc1,
  kNull,
  kNoop,
};

[[maybe_unused]]
__host__ __device__ bool isin(const char *chars, char c) {
  for (; *chars; chars++) {
    if (c == *chars) return true;
  }
  return false;
}

[[maybe_unused]]
__device__ __host__ uint32_t headpos(uint32_t b) {
  return b % (2 * kSingleTapeSize);
}

struct Bff {
  static const char *name();

#ifndef BFF_CUSTOM_OPS
  static __device__ __host__ BffOp GetOpKind(char c) {
    switch (c) {
      case '[':
        return BffOp::kLoopStart;
      case ']':
        return BffOp::kLoopEnd;
      case '+':
        return BffOp::kPlus;
      case '-':
        return BffOp::kMinus;
      case '.':
        return BffOp::kCopy01;
      case ',':
        return BffOp::kCopy10;
      case '<':
        return BffOp::kDec0;
      case '>':
        return BffOp::kInc0;
      case '{':
        return BffOp::kDec1;
      case '}':
        return BffOp::kInc1;
      case 0:
        return BffOp::kNull;
      default:
        return BffOp::kNoop;
    }
  }

#else
  static __device__ __host__ BffOp GetOpKind(char c);
#endif

  static __device__ __host__ const char *CommandRepr() { return "[]+-.,<>{}"; }

  static __device__ __host__ const char **CharacterRepr() {
    static const char *data[256] = {
        "\u0100", "\u0101", "\u0102", "\u0103", "\u0104", "\u0105", "\u0106",
        "\u0107", "\u0108", "\u0109", "\u010A", "\u010B", "\u010C", "\u010D",
        "\u010E", "\u010F", "\u0110", "\u0111", "\u0112", "\u0113", "\u0114",
        "\u0115", "\u0116", "\u0117", "\u0118", "\u0119", "\u011A", "\u011B",
        "\u011C", "\u011D", "\u011E", "\u011F", "\u0120", "\u0121", "\u0122",
        "\u0123", "\u0124", "\u0125", "\u0126", "\u0127", "\u0128", "\u0129",
        "\u012A", "\u012B", "\u012C", "\u012D", "\u012E", "\u012F", "\u0130",
        "\u0131", "\u0132", "\u0133", "\u0134", "\u0135", "\u0136", "\u0137",
        "\u0138", "\u0139", "\u013A", "\u013B", "\u013C", "\u013D", "\u013E",
        "\u013F", "\u0140", "\u0141", "\u0142", "\u0143", "\u0144", "\u0145",
        "\u0146", "\u0147", "\u0148", "\u0149", "\u014A", "\u014B", "\u014C",
        "\u014D", "\u014E", "\u014F", "\u0150", "\u0151", "\u0152", "\u0153",
        "\u0154", "\u0155", "\u0156", "\u0157", "\u0158", "\u0159", "\u015A",
        "\u015B", "\u015C", "\u015D", "\u015E", "\u015F", "\u0160", "\u0161",
        "\u0162", "\u0163", "\u0164", "\u0165", "\u0166", "\u0167", "\u0168",
        "\u0169", "\u016A", "\u016B", "\u016C", "\u016D", "\u016E", "\u016F",
        "\u0170", "\u0171", "\u0172", "\u0173", "\u0174", "\u0175", "\u0176",
        "\u0177", "\u0178", "\u0179", "\u017A", "\u017B", "\u017C", "\u017D",
        "\u017E", "\u017F", "\u0180", "\u0181", "\u0182", "\u0183", "\u0184",
        "\u0185", "\u0186", "\u0187", "\u0188", "\u0189", "\u018A", "\u018B",
        "\u018C", "\u018D", "\u018E", "\u018F", "\u0190", "\u0191", "\u0192",
        "\u0193", "\u0194", "\u0195", "\u0196", "\u0197", "\u0198", "\u0199",
        "\u019A", "\u019B", "\u019C", "\u019D", "\u019E", "\u019F", "\u01A0",
        "\u01A1", "\u01A2", "\u01A3", "\u01A4", "\u01A5", "\u01A6", "\u01A7",
        "\u01A8", "\u01A9", "\u01AA", "\u01AB", "\u01AC", "\u01AD", "\u01AE",
        "\u01AF", "\u01B0", "\u01B1", "\u01B2", "\u01B3", "\u01B4", "\u01B5",
        "\u01B6", "\u01B7", "\u01B8", "\u01B9", "\u01BA", "\u01BB", "\u01BC",
        "\u01BD", "\u01BE", "\u01BF", "\u01C0", "\u01C1", "\u01C2", "\u01C3",
        "A",      "B",      "C",      "D",      "E",      "F",      "G",
        "H",      "I",      "\u01CD", "\u01CE", "\u01CF", "\u01D0", "\u01D1",
        "\u01D2", "\u01D3", "\u01D4", "\u01D5", "\u01D6", "\u01D7", "\u01D8",
        "\u01D9", "\u01DA", "\u01DB", "\u01DC", "\u01DD", "\u01DE", "\u01DF",
        "\u01E0", "\u01E1", "\u01E2", "\u01E3", "\u01E4", "\u01E5", "\u01E6",
        "\u01E7", "\u01E8", "\u01E9", "\u01EA", "\u01EB", "\u01EC", "\u01ED",
        "\u01EE", "\u01EF", "\u01F0", "J",      "K",      "L",      "\u01F4",
        "\u01F5", "\u01F6", "\u01F7", "\u01F8", "\u01F9", "\u01FA", "\u01FB",
        "\u01FC", "\u01FD", "\u01FE", "\u01FF",
    };
    return data;
  }

  static void InitByteColors(
      std::array<std::array<uint8_t, 3>, 256> &byte_colors) {
    for (size_t i = 0; i < 256; i++) {
      const uint8_t v = 192 + i / 4;
      BffOp kind = GetOpKind(i);
      switch (kind) {
        case kLoopStart:
        case kLoopEnd:
          byte_colors[i] = {0, 192, 0};
          break;
        case kPlus:
        case kMinus:
        case kCopy01:
        case kCopy10:
          byte_colors[i] = {200, 0, 200};
          break;
        case kDec0:
        case kDec1:
        case kInc0:
        case kInc1:
          byte_colors[i] = {200, 128, 220};
          break;
        case kNull:
          byte_colors[i] = {255, 0, 0};
          break;
        default:
          byte_colors[i] = {v, v, v};
      }
    }
  }

  static std::string Parse(std::string bff) {
    std::string ret;
    char command_bytes[BffOp::kNoop] = {};
    for (size_t i = 0; i < 256; i++) {
      BffOp kind = GetOpKind(i);
      if (kind < BffOp::kNoop) {
        command_bytes[kind] = i;
      }
    }
    for (size_t i = 0; i < bff.size();) {
      if (bff[i] == '0') {
        ret.push_back(command_bytes[BffOp::kNull]);
        i += 1;
        continue;
      }
      for (size_t j = 0; j < 10; j++) {
        if (bff[i] == CommandRepr()[j]) {
          ret.push_back(command_bytes[j]);
          i += 1;
          goto found;
        }
      }
      for (size_t j = 0; j < 256; j++) {
        const char *s = CharacterRepr()[j];
        size_t l = strlen(s);
        if (bff.substr(i, l) == s) {
          ret.push_back(j);
          i += l;
          goto found;
        }
      }
      fprintf(stderr, "Invalid BFF program, character %d not recognized: %s\n",
              (int)i, bff.c_str());
    found:;
    }
    return ret;
  }

  static __device__ __host__ const char *MapChar(unsigned char c, char *mem) {
    BffOp kind = GetOpKind(c);
    bool is_command = kind < BffOp::kNull;
    if (is_command) {
      mem[0] = CommandRepr()[kind];
      mem[1] = 0;
      return mem;
    } else if (kind == BffOp::kNull) {
      return "0";
    } else {
      return CharacterRepr()[(int)c];
    }
  }

  static __device__ __host__ void PrintProgramInternal(
      size_t head0_pos, size_t head1_pos, size_t pc_pos, const uint8_t *mem,
      size_t len, const uint8_t *mem2, size_t len2) {
    auto print_char = [&](char c, size_t i) {
      BffOp kind = GetOpKind(c);
      bool is_command = kind < BffOp::kNull;
      if (kind == BffOp::kNull) {
        printf("\x1b[38:5:88m");
      } else if (is_command) {
        printf("\x1b[38:5:255m");
      } else {
        printf("\x1b[38:5:237m");
      }
      if (i == head0_pos) {
        printf("\x1b[48:5:27m");
      }
      if (i == head1_pos) {
        printf("\x1b[48:5:196m");
      }
      if (i == pc_pos) {
        printf("\x1b[48:5:34m");
      }
      char chmem[32] = {};
      printf("%s%s", MapChar(c, chmem), ResetColors());
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
    int head0_pos = 2 * kSingleTapeSize;
    int head1_pos = 2 * kSingleTapeSize;
    int fake_pc = 0;
    InitialState(mem, head0_pos, head1_pos, fake_pc);
    PrintProgramInternal(head0_pos, head1_pos, pc_pos, mem, len, mem2, len2);
  }

#ifdef BFF_HEADS
  static void __device__ __host__ InitialState(const uint8_t *tape, int &head0,
                                               int &head1, int &pc) {
    head0 = headpos(tape[0]);
    head1 = headpos(tape[1]);
    pc = 2;
  }
#else
  static void __device__ __host__ InitialState(const uint8_t *tape, int &head0,
                                               int &head1, int &pc) {
    head0 = 2 * kSingleTapeSize;
    head1 = 2 * kSingleTapeSize;
    pc = 0;
  }
#endif

  // Evaluation stops if `pc` is negative after this call. Otherwise, it is
  // incremented, and evaluation stops if it overflows 2*kSingleTapeSize.
  // Returns true if the op was not a comment.
#ifndef BFF_CUSTOM_LOGIC
  static bool __device__ EvaluateOne(uint8_t *tape, int &head0, int &head1,
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
        if (GetOpKind(tape[head0]) == BffOp::kNull) {
          size_t scanclosed = 1;
          pc++;
          for (; pc < (2 * kSingleTapeSize) && scanclosed > 0; pc++) {
            if (GetOpKind(tape[pc]) == BffOp::kLoopEnd) scanclosed--;
            if (GetOpKind(tape[pc]) == BffOp::kLoopStart) scanclosed++;
          }
          pc--;
          if (scanclosed != 0) {
            pc = 2 * kSingleTapeSize;
          }
        }
        break;
      case BffOp::kLoopEnd:
        if (GetOpKind(tape[head0]) != BffOp::kNull) {
          size_t scanopen = 1;
          pc--;
          for (; pc >= 0 && scanopen > 0; pc--) {
            if (GetOpKind(tape[pc]) == BffOp::kLoopEnd) scanopen++;
            if (GetOpKind(tape[pc]) == BffOp::kLoopStart) scanopen--;
          }
          pc++;
          if (scanopen != 0) {
            pc = -1;
          }
        }
        break;
      default:
        return false;
    }
    return true;
  }

#else
  static __device__ bool EvaluateOne(uint8_t *tape, int &head0, int &head1,
                                     int &pc);
#endif

  static __device__ size_t Evaluate(uint8_t *tape, size_t stepcount,
                                    bool debug) {
    size_t nskip = 0;

    int pos = 0;
    int head0_pos = 0;
    int head1_pos = 0;

    InitialState(tape, head0_pos, head1_pos, pos);

    size_t i = 0;
    for (; i < stepcount; i++) {
      head0_pos = head0_pos & (2 * kSingleTapeSize - 1);
      head1_pos = head1_pos & (2 * kSingleTapeSize - 1);
      if (debug) {
        PrintProgramInternal(head0_pos, head1_pos, pos, tape,
                             2 * kSingleTapeSize, nullptr, 0);
      }
      if (!EvaluateOne(tape, head0_pos, head1_pos, pos)) {
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
