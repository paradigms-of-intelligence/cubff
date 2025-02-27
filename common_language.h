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

#include <stdint.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#ifdef __CUDACC__
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define CUCHECK(op)                                                           \
  {                                                                           \
    cudaError_t cudaerr = op;                                                 \
    if (cudaerr != cudaSuccess) {                                             \
      printf("%s failed with error: %s\n", #op, cudaGetErrorString(cudaerr)); \
      exit(1);                                                                \
    }                                                                         \
  }

constexpr size_t kWarpSize = 32;

__inline__ __device__ uint64_t warpReduceSum(uint64_t val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(~0, val, offset);
  return val;
}

__inline__ __device__ size_t GetIndex() {
  return threadIdx.x + blockIdx.x * blockDim.x;
}

__inline__ __device__ void IncreaseInsnCount(unsigned long long count,
                                             unsigned long long *storage) {
  size_t index = GetIndex();
  size_t warp_ops = warpReduceSum(count);
  if (index % kWarpSize == 0) {
    atomicAdd(storage, warp_ops);
  }
}

inline void Synchronize() { CUCHECK(cudaDeviceSynchronize()); }

template <typename T>
struct DeviceMemory {
  T *data;
  DeviceMemory(size_t size) { CUCHECK(cudaMalloc(&data, size * sizeof(T))); }
  void Write(const T *host, size_t count) {
    CUCHECK(cudaMemcpy(data, host, count * sizeof(T), cudaMemcpyHostToDevice));
  }
  void Read(T *host, size_t count) {
    CUCHECK(cudaMemcpy(host, data, count * sizeof(T), cudaMemcpyDeviceToHost));
  }
  T *Get() { return data; }
  ~DeviceMemory() { CUCHECK(cudaFree(data)); }
  DeviceMemory(DeviceMemory &) = delete;
};

#define RUN(grid, block, fun, ...) fun<<<grid, block>>>(__VA_ARGS__)

#else
#define __device__
#define __host__
#define __global__

inline size_t &IndexThreadLocal() {
  thread_local size_t index;
  return index;
}

inline size_t GetIndex() { return IndexThreadLocal(); }

inline void IncreaseInsnCount(unsigned long long count,
                              unsigned long long *storage) {
  __atomic_add_fetch(storage, count, __ATOMIC_RELAXED);
}

inline void Synchronize() {}

template <typename T>
struct DeviceMemory {
  T *data;
  DeviceMemory(size_t size) { data = (T *)malloc(size * sizeof(T)); }
  void Write(const T *host, size_t count) {
    memcpy(data, host, count * sizeof(T));
  }
  void Read(T *host, size_t count) { memcpy(host, data, count * sizeof(T)); }
  T *Get() { return data; }
  ~DeviceMemory() { free(data); }
  DeviceMemory(DeviceMemory &) = delete;
};

#define RUN(grid, block, fun, ...)                                            \
  _Pragma("omp parallel for") for (size_t _threadcnt = 0;                     \
                                   _threadcnt < grid * block; _threadcnt++) { \
    IndexThreadLocal() = _threadcnt;                                          \
    fun(__VA_ARGS__);                                                         \
  }

#endif

#define CHECK(op)                 \
  if (!(op)) {                    \
    printf("%s is false\n", #op); \
    exit(1);                      \
  }

inline __device__ __host__ uint64_t SplitMix64(uint64_t seed) {
  uint64_t z = seed + 0x9e3779b97f4a7c15;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

template <typename Language>
__global__ void InitPrograms(size_t seed, size_t num_programs,
                             uint8_t *programs, bool zero_init) {
  size_t index = GetIndex();
  auto prog = programs + index * kSingleTapeSize;
  if (index >= num_programs) return;
  if (zero_init) {
    for (size_t i = 0; i < kSingleTapeSize; i++) {
      prog[i] = 0;
    }
  } else {
    for (size_t i = 0; i < kSingleTapeSize; i++) {
      prog[i] = SplitMix64(kSingleTapeSize * num_programs * seed +
                           kSingleTapeSize * index + i) %
                256;
    }
  }
}

template <typename Language>
__global__ void MutateAndRunPrograms(uint8_t *programs,
                                     const uint32_t *shuf_idx, size_t seed,
                                     uint32_t mutation_prob,
                                     unsigned long long *insn_count,
                                     size_t num_programs, size_t num_indices) {
  size_t index = GetIndex();
  uint8_t tape[2 * kSingleTapeSize] = {};
  if (2 * index >= num_programs) return;
  uint32_t p1 = shuf_idx[2 * index];
  uint32_t p2 = shuf_idx[2 * index + 1];
  for (size_t i = 0; i < kSingleTapeSize; i++) {
    tape[i] = programs[p1 * kSingleTapeSize + i];
    tape[i + kSingleTapeSize] = programs[p2 * kSingleTapeSize + i];
  }
  for (size_t i = 0; i < 2 * kSingleTapeSize; i++) {
    uint64_t rng =
        SplitMix64((num_programs * seed + index) * kSingleTapeSize * 2 + i);
    uint8_t repl = rng & 0xFF;
    uint64_t prob_rng = (rng >> 8) & ((1ULL << 30) - 1);
    if (prob_rng < mutation_prob) {
      tape[i] = repl;
    }
  }
  bool debug = false;
  size_t ops;
  if (index < num_indices) {
    ops = Language::Evaluate(tape, 8 * 1024, debug);
  } else {
    ops = 0;
  }
  for (size_t i = 0; i < kSingleTapeSize; i++) {
    programs[p1 * kSingleTapeSize + i] = tape[i];
    programs[p2 * kSingleTapeSize + i] = tape[i + kSingleTapeSize];
  }
  IncreaseInsnCount(ops, insn_count);
}

template <typename Language>
__global__ void RunOneProgram(uint8_t *program, size_t stepcount, bool debug) {
  size_t ops = Language::Evaluate(program, stepcount, debug);
  printf("%s", ResetColors());
  printf("ops: %d\n", (int)ops);
  printf("\n");
}

template <typename Language>
__global__ void CheckSelfRep(uint8_t *programs, size_t seed,
                             size_t num_programs, size_t *result, bool debug) {
  size_t index = GetIndex();
  constexpr size_t kNumIters = 13;
  constexpr size_t kNumExtraGens = 4;
  uint8_t tapes[kNumIters][2 * kSingleTapeSize] = {};
  if (index > num_programs) return;
  uint64_t local_seed = SplitMix64(num_programs * seed + index);
  for (size_t i = 0; i < kNumIters; i++) {
    bool eval_debug = false;
    uint8_t noise[kSingleTapeSize];
    for (int j = 0; j < kSingleTapeSize; j++) {
      noise[j] =
          SplitMix64(local_seed ^ SplitMix64((i + 1) * kSingleTapeSize + j)) %
          256;
    }
    uint8_t *tape = &tapes[i][0];
    for (int j = 0; j < kSingleTapeSize; j++) {
      tape[j] = programs[index * kSingleTapeSize + j];
      tape[j + kSingleTapeSize] = noise[j];
    }
    if (debug) {
      size_t separators[1] = {kSingleTapeSize};
      printf("Iteration %lu: before first step\n", i);
      Language::PrintProgram(2 * kSingleTapeSize, tape, 2 * kSingleTapeSize,
                             separators, 1);
    }
    Language::Evaluate(tape, 8 * 1024, eval_debug);
    if (debug) {
      size_t separators[1] = {kSingleTapeSize};
      printf("Iteration %lu: after first step\n", i);
      Language::PrintProgram(2 * kSingleTapeSize, tape, 2 * kSingleTapeSize,
                             separators, 1);
    }

    for (size_t g = 0; g < kNumExtraGens; g++) {
      for (int j = 0; j < kSingleTapeSize; j++) {
        tape[j] = tape[j + kSingleTapeSize];
        tape[j + kSingleTapeSize] = noise[j];
      }
      if (debug) {
        size_t separators[1] = {kSingleTapeSize};
        printf("Iteration %lu: before step %lu\n", i, g + 2);
        Language::PrintProgram(2 * kSingleTapeSize, tape, 2 * kSingleTapeSize,
                               separators, 1);
      }
      Language::Evaluate(tape, 8 * 1024, eval_debug);
      if (debug) {
        size_t separators[1] = {kSingleTapeSize};
        printf("Iteration %lu: after step %lu\n", i, g + 2);
        Language::PrintProgram(2 * kSingleTapeSize, tape, 2 * kSingleTapeSize,
                               separators, 1);
      }
    }
  }
  size_t res[2] = {};
  for (int i = 0; i < 2 * kSingleTapeSize; ++i) {
    for (size_t a = 0; a < kNumIters; a++) {
      size_t count = 1;
      if (i < kSingleTapeSize &&
          tapes[a][i] != programs[index * kSingleTapeSize + i]) {
        continue;
      }
      for (size_t b = a + 1; b < kNumIters; b++) {
        if (tapes[a][i] == tapes[b][i]) count++;
      }
      if (count > kNumIters / 4) {
        res[i / kSingleTapeSize]++;
        break;
      }
    }
  }
  result[index] = res[0] < res[1] ? res[0] : res[1];
}

template <typename Language>
void Simulation<Language>::RunSingleParsedProgram(
    const std::vector<uint8_t> &parsed, size_t stepcount, bool debug) const {
  DeviceMemory<uint8_t> mem(kSingleTapeSize * 2);
  uint8_t zero[2 * kSingleTapeSize] = {};
  memcpy(zero, parsed.data(), parsed.size());
  mem.Write(zero, 2 * kSingleTapeSize);
  Language::PrintProgram(2 * kSingleTapeSize, zero, 2 * kSingleTapeSize,
                         nullptr, 0);

  RUN(1, 1, RunOneProgram<Language>, mem.Get(), stepcount, debug);

  uint8_t final_state[2 * kSingleTapeSize];
  Synchronize();
  mem.Read(final_state, 2 * kSingleTapeSize);
  Language::PrintProgram(2 * kSingleTapeSize, final_state, 2 * kSingleTapeSize,
                         nullptr, 0);
}

template <typename Language>
void Simulation<Language>::RunSingleProgram(std::string program,
                                            size_t stepcount,
                                            bool debug) const {
  RunSingleParsedProgram(Language::Parse(program), stepcount, debug);
}

template <typename Language>
void Simulation<Language>::PrintProgram(size_t pc_pos, const uint8_t *mem,
                                        size_t len, const size_t *separators,
                                        size_t num_separators) const {
  Language::PrintProgram(pc_pos, mem, len, separators, num_separators);
}

template <typename Language>
size_t Simulation<Language>::EvalSelfrep(std::string program, size_t epoch,
                                         size_t seed, bool debug) {
  std::vector<uint8_t> parsed = Language::Parse(program);
  return EvalParsedSelfrep(parsed, epoch, seed, debug);
}

template <typename Language>
size_t Simulation<Language>::EvalParsedSelfrep(std::vector<uint8_t> &parsed,
                                               size_t epoch, size_t seed,
                                               bool debug) {
  DeviceMemory<uint8_t> mem(kSingleTapeSize);
  uint8_t zero[kSingleTapeSize] = {};
  memcpy(zero, parsed.data(), parsed.size());
  mem.Write(zero, kSingleTapeSize);
  DeviceMemory<size_t> result(1);
  size_t epoch_seed = SplitMix64(SplitMix64(seed) ^ SplitMix64(epoch));
  RUN(1, 1, CheckSelfRep<Language>, mem.Get(), epoch_seed, 1, result.Get(),
      debug);

  Synchronize();
  std::vector<size_t> res(1);
  result.Read(res.data(), 1);
  return res[0];
}

template <typename Language>
void Simulation<Language>::RunSimulation(
    const SimulationParams &params, std::optional<std::string> initial_program,
    std::function<bool(const SimulationState &)> callback) const {
  constexpr size_t kNumThreads = 32;
  size_t num_programs = params.num_programs;

  size_t reset_index = 1;
  size_t epoch = 0;

  FILE *load_file = nullptr;
  if (params.load_from.has_value()) {
    load_file = CheckFopen(params.load_from->c_str(), "r");
    CHECK(fread(&reset_index, sizeof(reset_index), 1, load_file) == 1);
    CHECK(fread(&num_programs, sizeof(num_programs), 1, load_file) == 1);
    CHECK(fread(&epoch, sizeof(epoch), 1, load_file) == 1);
  }

  DeviceMemory<uint8_t> programs(kSingleTapeSize * num_programs);
  DeviceMemory<unsigned long long> insn_count(1);

  CHECK(num_programs % 2 == 0);

  auto seed = [&](size_t seed2) {
    return SplitMix64(SplitMix64(params.seed) ^ SplitMix64(seed2));
  };

  RUN((num_programs + kNumThreads - 1) / kNumThreads, kNumThreads,
      InitPrograms<Language>, seed(0), num_programs, programs.Get(),
      params.zero_init);

  if (initial_program.has_value()) {
    std::vector<uint8_t> parsed = Language::Parse(*initial_program);
    programs.Write((const unsigned char *)parsed.data(), parsed.size());
  }

  unsigned long long zero = 0;
  insn_count.Write(&zero, 1);

  unsigned long long total_ops = 0;

  SimulationState state;
  state.soup.reserve(num_programs * kSingleTapeSize + 16);
  state.soup.resize(num_programs * kSingleTapeSize);
  state.replication_per_prog.resize(num_programs);
  state.shuffle_idx.resize(num_programs);
  Language::InitByteColors(state.byte_colors);

  if (params.save_to.has_value()) {
    CHECK(mkdir(params.save_to->c_str(),
                S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != -1 ||
          errno == EEXIST);
  }

  if (load_file) {
    CHECK(fread(state.soup.data(), 1, num_programs * kSingleTapeSize,
                load_file) == num_programs * kSingleTapeSize);
    fclose(load_file);
    programs.Write(state.soup.data(), num_programs * kSingleTapeSize);
  }

  DeviceMemory<uint32_t> shuf_idx(num_programs);

  std::vector<uint32_t> &s = state.shuffle_idx;

  for (size_t i = 0; i < num_programs; i++) {
    s[i] = i;
  }

  std::vector<uint32_t> shuffle_tmp_buf(num_programs);
  std::vector<char> used_program(num_programs);

  Synchronize();

  auto do_shuffle = [&](uint32_t *begin, uint32_t *end, uint64_t base_seed) {
    size_t len = end - begin;
    for (size_t i = len; i-- > 0;) {
      size_t j = SplitMix64(seed(epoch * len + i)) % (i + 1);
      std::swap(begin[i], begin[j]);
    }
  };

  std::vector<uint8_t> brotlified_data(
      BrotliEncoderMaxCompressedSize(num_programs * kSingleTapeSize));

  size_t num_runs = 0;
  auto start = std::chrono::high_resolution_clock::now();
  auto simulation_start = std::chrono::high_resolution_clock::now();
  for (;; epoch++) {
    size_t num_indices = num_programs / 2;
    // Shuffle indices.
    if (!params.allowed_interactions.empty()) {
      for (size_t i = 0; i < num_programs; i++) {
        shuffle_tmp_buf[i] = i;
        used_program[i] = false;
      }
      do_shuffle(shuffle_tmp_buf.data(),
                 shuffle_tmp_buf.data() + shuffle_tmp_buf.size(), epoch);
      num_indices = 0;
      for (size_t i : shuffle_tmp_buf) {
        auto &interact = params.allowed_interactions;
        if (interact.size() <= i || interact[i].empty()) {
          continue;
        }
        size_t idx = seed(seed(epoch) ^ seed(i)) % interact[i].size();
        size_t neigh = interact[i][idx];
        if (used_program[i] || used_program[neigh]) {
          continue;
        }
        used_program[i] = used_program[neigh] = true;
        s[num_indices * 2] = i;
        s[num_indices * 2 + 1] = neigh;
        num_indices++;
      }
      size_t idx = num_indices * 2;
      for (size_t i = 0; i < num_programs; i++) {
        if (!used_program[i]) {
          s[idx++] = i;
        }
      }
    } else if (params.permute_programs) {
      for (size_t i = 0; i < num_programs; i++) {
        s[i] = i;
      }
      if (params.fixed_shuffle) {
        size_t flip = epoch & 1;
        size_t max_pow2 = 31 - __builtin_clz(num_programs);
        size_t offset = (1 << (epoch % max_pow2 + 1)) - 1;
        for (size_t i = 0; i < num_programs; i++) {
          s[i] = ((i * offset) % num_programs) ^ flip;
        }
      } else {
        do_shuffle(s.data(), s.data() + s.size(), epoch);
      }
    } else if (epoch % 2 == 1) {
      for (size_t i = 0; i < num_programs; i++) {
        s[i] = i;
      }
    } else {
      for (size_t i = 0; i < num_programs; i++) {
        s[i] = i == 0 ? num_programs - 1 : i - 1;
      }
    }

    shuf_idx.Write(s.data(), num_programs);

    RUN((num_programs + 2 * kNumThreads - 1) / (2 * kNumThreads), kNumThreads,
        MutateAndRunPrograms<Language>, programs.Get(), shuf_idx.Get(),
        seed(epoch), params.mutation_prob, insn_count.Get(), num_programs,
        num_indices);
    num_runs += num_indices;

    if (epoch % params.callback_interval == 0) {
      auto stop = std::chrono::high_resolution_clock::now();
      Synchronize();
      unsigned long long insn;
      insn_count.Read(&insn, 1);
      total_ops += insn;
      programs.Read(state.soup.data(), num_programs * kSingleTapeSize);
      Synchronize();
      size_t brotli_size = brotlified_data.size();
      BrotliEncoderCompress(2, 24, BROTLI_MODE_GENERIC, state.soup.size(),
                            state.soup.data(), &brotli_size,
                            brotlified_data.data());
      float elapsed_s =
          std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
              .count() *
          1e-6;
      float mops_s = insn * 1.0 / elapsed_s * 1e-6;
      float sim_elapsed_s =
          std::chrono::duration_cast<std::chrono::microseconds>(
              stop - simulation_start)
              .count() *
          1e-6;

      size_t counts[256] = {};
      for (auto c : state.soup) {
        counts[c]++;
      }

      std::vector<uint8_t> sorted(256);
      double h0 = 0;
      for (size_t i = 0; i < 256; i++) {
        sorted[i] = i;
        double frac = counts[i] * 1.0 / state.soup.size();
        h0 -= counts[i] ? frac * std::log2(frac) : 0.0;
      }
      std::sort(sorted.begin(), sorted.end(), [&](uint8_t a, uint8_t b) {
        return std::make_pair(counts[b], b) < std::make_pair(counts[a], a);
      });

      double brotli_bpb = brotli_size * 8.0 / (num_programs * kSingleTapeSize);

      state.elapsed_s = sim_elapsed_s;
      state.total_ops = total_ops;
      state.mops_s = mops_s;
      state.epoch = epoch + 1;
      state.ops_per_run = insn * 1.0 / num_runs;
      state.brotli_size = brotli_size;
      state.brotli_bpb = brotli_bpb;
      state.bytes_per_prog = brotli_size * 1.0 / num_programs;
      state.h0 = h0;
      state.higher_entropy = h0 - brotli_bpb;

      for (size_t i = 0; i < state.frequent_bytes.size(); i++) {
        uint8_t c = sorted[i];
        char chmem[32];
        state.frequent_bytes[i].first = Language::MapChar(c, chmem);
        state.frequent_bytes[i].second =
            counts[(int)c] * 1.0 / state.soup.size();
      }
      for (size_t i = 0; i < state.uncommon_bytes.size(); i++) {
        uint8_t c = sorted[256 - state.uncommon_bytes.size() + i];
        char chmem[32];
        state.uncommon_bytes[i].first = Language::MapChar(c, chmem);
        state.uncommon_bytes[i].second =
            counts[(int)c] * 1.0 / state.soup.size();
      }

      if (params.eval_selfrep) {
        DeviceMemory<size_t> result(num_programs);
        RUN(num_programs / kNumThreads, kNumThreads, CheckSelfRep<Language>,
            programs.Get(), seed(epoch), num_programs, result.Get(), false);
        Synchronize();
        result.Read(state.replication_per_prog.data(), num_programs);
      }
      if (params.save_to.has_value() && (epoch % params.save_interval == 0)) {
        std::vector<char> save_path(params.save_to->size() + 20);
        snprintf(save_path.data(), save_path.size(), "%s/%010zu.dat",
                 params.save_to->c_str(), epoch);
        FILE *f = CheckFopen(save_path.data(), "w");
        size_t epoch_to_save = epoch + 1;
        fwrite(&reset_index, sizeof(reset_index), 1, f);
        fwrite(&num_programs, sizeof(num_programs), 1, f);
        fwrite(&epoch_to_save, sizeof(epoch), 1, f);
        fwrite(state.soup.data(), 1, state.soup.size(), f);
        fclose(f);
      }
      if (callback(state)) {
        break;
      }
      num_runs = 0;
      start = std::chrono::high_resolution_clock::now();
      insn_count.Write(&zero, 1);
    }

    if (params.reset_interval.has_value() &&
        epoch % *params.reset_interval == 0) {
      RUN(num_programs / kNumThreads, kNumThreads, InitPrograms<Language>,
          seed(reset_index), num_programs, programs.Get(), params.zero_init);
      reset_index++;
    }
  }
}
