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

#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "brotli/encode.h"

#define REGISTER(L)                                               \
  static void f() __attribute__((constructor));                   \
  static void f() {                                               \
    RegisterLanguage(L::name(), &Simulation<L>::RunSingleProgram, \
                     &Simulation<L>::RunSimulation);              \
  }

#ifdef __CUDACC__
__device__ __host__
#endif
    inline constexpr const char *
    ResetColors() {
  return "\x1b[0;38:5:35;48:5:232m";
}

constexpr int kSingleTapeSize = 64;

struct SimulationParams {
  size_t num_programs = 128 * 1024;
  size_t seed = 0;
  uint32_t mutation_prob = 1 << 18;  // denominator 1<<30.
  std::optional<size_t> reset_interval = std::nullopt;
  std::optional<std::string> load_from = std::nullopt;
  std::optional<std::string> save_to = std::nullopt;
  size_t callback_interval = 128;
  size_t save_interval = 0;
  bool permute_programs = true;
  bool fixed_shuffle = false;
  bool zero_init = false;
  std::vector<std::vector<uint32_t>> allowed_interactions;
};

struct SimulationState {
  std::vector<uint8_t> soup;
  std::vector<uint32_t> shuffle_idx;
  std::array<std::array<uint8_t, 3>, 256> byte_colors;
  float elapsed_s;
  size_t total_ops;
  float mops_s;
  size_t epoch;
  float ops_per_run;
  size_t brotli_size;
  float brotli_bpb;
  float bytes_per_prog;
  float h0;
  float higher_entropy;
  std::array<std::pair<std::string, float>, 16> frequent_bytes;
  std::array<std::pair<std::string, float>, 16> uncommon_bytes;
  std::function<void(size_t)> print_program;
};

template <typename Language>
struct Simulation {
  static void RunSingleProgram(std::string program, size_t stepcount,
                               bool debug);
  static void RunSimulation(
      const SimulationParams &params,
      std::optional<std::string> initial_program,
      std::function<bool(const SimulationState &)> callback);
};

using runsingle_t = void (*)(std::string, size_t, bool);
using runsimulation_t = void (*)(const SimulationParams &,
                                 std::optional<std::string>,
                                 std::function<bool(const SimulationState &)>);

void RegisterLanguage(const char *lang, runsingle_t runsingle,
                      runsimulation_t runsimulation);

template <typename Language>
void Register() {}

void RunSingleProgram(const std::string &language, std::string program,
                      size_t stepcount, bool debug);
void RunSimulation(const std::string &language, const SimulationParams &params,
                   std::optional<std::string> initial_program,
                   std::function<bool(const SimulationState &)> callback);

inline FILE *CheckFopen(const char *f, const char *mode) {
  FILE *out = fopen(f, mode);
  char buf[4096];
  if (out == nullptr) {
    fprintf(stderr, "Could not open %s: %s\n", f,
            strerror_r(errno, buf, sizeof(buf)));
    exit(1);
  }
  return out;
}
