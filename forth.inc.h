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

enum ForthOp {
  kWrite0,
  kWrite1,
  kWrite,
  kRead,
  kRead0,
  kRead1,
  kDup,
  kDrop,
  kSwap,
  kIf0,
  kInc,
  kDec,
  kAdd,
  kSub,
  kCopy,
  kCopy0,
  kCopy1,
  kXor,
  kConst,
  kJmp,
  kNoop,
};

struct Forth {
  static const char *name();

#ifndef FORTH_CUSTOM_OPS
  static __device__ __host__ ForthOp GetOpKind(uint8_t c) {
    switch (c) {
      case 0x0:
        return ForthOp::kRead0;
      case 0x1:
        return ForthOp::kRead1;
      case 0x2:
        return ForthOp::kWrite0;
      case 0x3:
        return ForthOp::kWrite1;
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

#else
  static __device__ __host__ ForthOp GetOpKind(uint8_t c);
#endif

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
    auto print_char = [&](char c, int i) {
      ForthOp kind = GetOpKind((uint8_t)c);
      char chmem[4];
      if (i == pc_pos) {
        printf("\x1b[48:5:22m");
      }
      if (kind == kJmp) {
        ((uint8_t)c) & 64
            ? printf("\x1b[38:5:221m\e]8;;%d%s%s%d\e\\%s\e]8;;\e\\", i, "Jmp",
                     "-", (((uint8_t)c) & 63) + 1, MapChar(c, chmem))
            : printf("\x1b[38:5:117m\e]8;;%d%s%s%d\e\\%s\e]8;;\e\\", i, "Jmp",
                     "+", (((uint8_t)c) & 63) + 1, MapChar(c, chmem));
      } else if (kind == kConst) {
        printf("\x1b[38:5:255m\e]8;;%d%s%d\e\\%s\e]8;;\e\\", i, "Const",
               ((uint8_t)c) & 63, MapChar(c, chmem));
      } else {
        if (i % 16 == 0) {
          // printf("\e]8;;%d\e\\", i);
        }

        switch (kind) {
          case kWrite:
            printf("\x1b[38:5:207m%s", MapChar(c, chmem));
            break;
          case kWrite0:
            printf("\x1b[38:5:207m%s", MapChar(c, chmem));
            break;
          case kWrite1:
            printf("\x1b[38:5:207m%s", MapChar(c, chmem));
            break;
          case kRead:
            printf("\x1b[38:5:112m%s", MapChar(c, chmem));
            break;
          case kRead0:
            printf("\x1b[38:5:112m%s", MapChar(c, chmem));
            break;
          case kRead1:
            printf("\x1b[38:5:112m%s", MapChar(c, chmem));
            break;
          case kCopy:
            printf("\x1b[38:5:141m%s", MapChar(c, chmem));
            break;
          case kCopy0:
            printf("\x1b[38:5:141m%s", MapChar(c, chmem));
            break;
          case kCopy1:
            printf("\x1b[38:5:141m%s", MapChar(c, chmem));
            break;
          case kXor:
            printf("\x1b[38:5:196m%s", MapChar(c, chmem));
            break;
          case kDup:
            printf("\x1b[38:5:255m%s", MapChar(c, chmem));
            break;
          case kDrop:
            printf("\x1b[38:5:255m%s", MapChar(c, chmem));
            break;
          case kSwap:
            printf("\x1b[38:5:255m%s", MapChar(c, chmem));
            break;
          case kIf0:
            printf("\x1b[38:5:255m%s", MapChar(c, chmem));
            break;
          case kInc:
            printf("\x1b[38:5:255m%s", MapChar(c, chmem));
            break;
          case kDec:
            printf("\x1b[38:5:255m%s", MapChar(c, chmem));
            break;
          case kAdd:
            printf("\x1b[38:5:255m%s", MapChar(c, chmem));
            break;
          case kSub:
            printf("\x1b[38:5:255m%s", MapChar(c, chmem));
            break;
          case kNoop:
            printf("\x1b[38:5:237m%s", MapChar(c, chmem));
            break;
          default:
            break;
        }
      }
      if (i % 16 == 0) {
        // printf("\e]8;;\e\\");
      }
      printf("%s", ResetColors());
    };
    size_t sep_id = 0;
    for (int i = 0; i < len; i++) {
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
#ifndef FORTH_CUSTOM_LOGIC
  static __device__ void EvaluateOne(uint8_t *tape, int &pos, size_t &nops,
                                     Stack &stack) {
    uint8_t command = tape[pos];
    switch (GetOpKind(command)) {
      case kRead0: {
        int addr = stack.Pop() % kSingleTapeSize;
        stack.Push(tape[0 + addr]);
        break;
      }
      case kRead1: {
        int addr = stack.Pop() % kSingleTapeSize;
        stack.Push(tape[kSingleTapeSize + addr]);
        break;
      }
      case kRead: {
        int addr = stack.Pop() % (2 * kSingleTapeSize);
        stack.Push(tape[addr]);
        break;
      }
      case kWrite: {
        int val = stack.Pop();
        int addr = stack.Pop() % (2 * kSingleTapeSize);
        tape[addr] = val;
        break;
      }

      case kWrite0: {
        int val = stack.Pop();
        int addr = stack.Pop() % kSingleTapeSize;
        tape[0 + addr] = val;
        break;
      }
      case kWrite1: {
        int val = stack.Pop();
        int addr = stack.Pop() % kSingleTapeSize;
        tape[kSingleTapeSize + addr] = val;
        break;
      }
      case kDup: {
        int v = stack.Pop();
        stack.Push(v);
        stack.Push(v);
        break;
      }
      case kDrop:
        stack.Pop();
        break;
      case kSwap: {
        int a = stack.Pop();
        int b = stack.Pop();
        stack.Push(a);
        stack.Push(b);
        break;
      }
      case kIf0: {
        int v = stack.Pop();
        if (v) {
          pos++;
        }
        stack.Push(v);
        break;
      }
      case kInc: {
        stack.Push(stack.Pop() + 1);
        break;
      }
      case kDec: {
        stack.Push(stack.Pop() - 1);
        break;
      }
      case kAdd: {
        int a = stack.Pop();
        int b = stack.Pop();
        stack.Push(a + b);
        break;
      }
      case kSub: {
        int a = stack.Pop();
        int b = stack.Pop();
        stack.Push(a - b);
        break;
      }
      case kConst: {
        // 01xxxxxx (40-7F) -> stack.Push unsigned constant xxxxxx
        stack.Push(command & 63);
        break;
      }
      case kCopy0: {
        int addr = stack.Pop() % kSingleTapeSize;
        tape[kSingleTapeSize + addr] = tape[0 + addr];
        break;
      }
      case kCopy1: {
        int addr = stack.Pop() % kSingleTapeSize;
        tape[0 + addr] = tape[kSingleTapeSize + addr];
        break;
      }
      case kCopy: {
        int to = stack.Pop() % (2 * kSingleTapeSize);
        int from = stack.Pop() % (2 * kSingleTapeSize);
        tape[to] = tape[from];
        break;
      }
      case kXor: {
        stack.Push(stack.Pop() ^ 64);
        break;
      }

      case kJmp: {
        // 1Xxxxxxx (80-FF) -> jump to offset {+-}(xxxxxx+1)
        int abs = (command & 63) + 1;
        int jmp = command & 64 ? -abs : abs;
        pos += jmp;
        pos--;
        break;
      }

      default: {
        nops++;
      }
    }
    pos++;
  }
#else

  static __device__ void EvaluateOne(uint8_t *tape, int &pos, size_t &nops,
                                     Stack &stack);
#endif
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
