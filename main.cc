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

#include <assert.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "common.h"

namespace flags {
class BaseFlag {
 public:
  static std::unordered_map<std::string, BaseFlag *> *reg() {
    static std::unordered_map<std::string, BaseFlag *> r;
    return &r;
  }

  virtual size_t Parse(int argc, char **argv) = 0;
  virtual const char *Description() = 0;
  virtual void PrintValue() = 0;
  virtual bool ShouldPrintDescription(const std::string &flag) = 0;

  virtual ~BaseFlag() {}

  static void ParseCommandLine(int argc, char **argv) {
    auto help_and_exit = [&]() {
      fprintf(stderr, "%s flags\n\n", argv[0]);
      std::vector<std::string> all_flags;
      for (const auto &[f, _] : *reg()) {
        all_flags.push_back(f);
      }
      std::sort(all_flags.begin(), all_flags.end());
      for (const auto &f : all_flags) {
        if (!(*reg())[f]->ShouldPrintDescription(f)) {
          continue;
        }
        fprintf(stderr, "     %s: %s; default: ", f.c_str(),
                (*reg())[f]->Description());
        (*reg())[f]->PrintValue();
        fprintf(stderr, "\n");
      }
      std::exit(1);
    };

    for (int pos = 1; pos < argc; pos++) {
      if (argv[pos] == std::string("--help") ||
          argv[pos] == std::string("-h")) {
        help_and_exit();
      }

      auto iter = (*reg()).find(argv[pos]);
      if (iter == (*reg()).end()) {
        fprintf(stderr, "Unknown flag %s\n", argv[pos]);
        help_and_exit();
      }
      BaseFlag *parser = iter->second;
      pos += parser->Parse(argc - pos, argv + pos);
    }
  }
};

template <typename T>
struct Types;

template <>
struct Types<std::string> {
  static std::string Parse(const char *v) { return v; }
  static void Print(const std::string &v) { fprintf(stderr, "%s", v.c_str()); }
};

template <>
struct Types<double> {
  static double Parse(const char *v) { return std::stod(v); }
  static void Print(const double &v) { fprintf(stderr, "%8.5f", v); }
};

template <>
struct Types<size_t> {
  static size_t Parse(const char *v) { return std::stoul(v); }
  static void Print(const size_t &v) { fprintf(stderr, "%zu", v); }
};

template <typename T>
struct Types<std::optional<T>> {
  static std::optional<T> Parse(const char *v) { return Types<T>::Parse(v); }
  static void Print(const std::optional<T> &v) {
    if (v.has_value()) {
      Types<T>::Print(*v);
    } else {
      fprintf(stderr, "<no value>");
    }
  }
};

template <typename T>
class Flag : public BaseFlag {
 public:
  Flag(const char *opt, const char *noopt, T default_value,
       const char *description)
      : opt_(opt),
        noopt_(noopt),
        value_(std::move(default_value)),
        description_(description) {
    (*reg())[opt] = this;
    if constexpr (std::is_same_v<bool, T>) {
      (*reg())[noopt] = this;
    }
  }

  const T &Get() const { return value_; }

  size_t Parse(int argc, char **argv) override {
    if constexpr (std::is_same_v<bool, T>) {
      value_ = (argv[0] == std::string(opt_));
      return 0;
    } else {
      if (argc < 2) {
        fprintf(stderr, "missing argument for flag %s\n", argv[0]);
        std::exit(1);
      }
      value_ = Types<T>::Parse(argv[1]);
      return 1;
    }
  }
  const char *Description() override { return description_; }
  bool ShouldPrintDescription(const std::string &flag) override {
    return flag == opt_;
  }
  void PrintValue() override {
    if constexpr (std::is_same_v<bool, T>) {
      fprintf(stderr, "%s", value_ ? "true" : "false");
    } else {
      Types<T>::Print(value_);
    }
  }

 private:
  const char *opt_;
  const char *noopt_;
  T value_;
  const char *description_;
};

#define FLAG(type, name, default_value, description)                      \
  flags::Flag<type> FLAGS_##name("--" #name, "--no" #name, default_value, \
                                 description);

template <typename T>
const T &GetFlag(const Flag<T> &flag) {
  return flag.Get();
}

void ParseCommandLine(int argc, char **argv) {
  BaseFlag::ParseCommandLine(argc, argv);
}

}  // namespace flags

FLAG(std::optional<std::string>, run, std::nullopt, "run a program");
FLAG(size_t, run_steps, 32 * 1024, "max number of steps for running a program");
FLAG(bool, debug, false, "print execution step by step");
FLAG(size_t, num, 128 * 1024, "number of programs to evolve");
FLAG(std::optional<size_t>, max_epochs, std::nullopt, "max epochs");
FLAG(size_t, seed, 0, "seed");
FLAG(double, mutation_prob, 1.0 / (256 * 16), "mutation_prob");
FLAG(std::optional<double>, stopping_bpb, std::nullopt,
     "bits per byte below which to stop execution");
FLAG(std::optional<std::string>, initial_program, std::nullopt,
     "program to seed the soup with");
FLAG(std::optional<std::string>, log, std::nullopt, "log file");
FLAG(std::optional<std::string>, load, std::nullopt, "load a previous save");
FLAG(std::optional<std::string>, checkpoint_dir, std::nullopt,
     "directory to store checkpoints");
FLAG(std::optional<size_t>, reset_interval, std::nullopt, "reset interval");
FLAG(std::string, lang, "", "language to run");
FLAG(std::string, interaction_pattern, "",
     "file containing the allowed interactions, with two integers `a b` per "
     "line, representing that program `a` is allowed to interact with `b` (in "
     "that order).");
FLAG(bool, permute_programs, true,
     "do not shuffle programs between runs (cyclic interactions)");
FLAG(bool, fixed_shuffle, false, "deterministic shuffling pattern");
FLAG(bool, zero_init, false, "zero init");
FLAG(size_t, print_interval, 64, "interval between prints");
FLAG(size_t, save_interval, 256, "interval between saves");
FLAG(size_t, clear_interval, 2048, "interval between clears");
FLAG(std::string, draw_to, "",
     "directory to save 1d-drawn frames to (must exist)");
FLAG(std::string, draw_to_2d, "",
     "directory to save 2d-drawn frames to (must exist, and num must "
     "be a square number)");
FLAG(size_t, grid_width_2d, 0, "width of the 2d grid");
FLAG(bool, disable_output, false, "disable printing to stdout");

int main(int argc, char **argv) {
  flags::ParseCommandLine(argc, argv);

  bool debug = GetFlag(FLAGS_debug);

  SimulationParams params;
  params.num_programs = GetFlag(FLAGS_num);
  params.reset_interval = GetFlag(FLAGS_reset_interval);
  params.seed = GetFlag(FLAGS_seed);
  params.load_from = GetFlag(FLAGS_load);
  params.mutation_prob = std::round(GetFlag(FLAGS_mutation_prob) * (1 << 30));
  params.permute_programs = GetFlag(FLAGS_permute_programs);
  params.fixed_shuffle = GetFlag(FLAGS_fixed_shuffle);
  params.zero_init = GetFlag(FLAGS_zero_init);
  params.save_to = GetFlag(FLAGS_checkpoint_dir);
  params.save_interval = GetFlag(FLAGS_save_interval);
  if (params.fixed_shuffle &&
      (params.num_programs & (params.num_programs - 1)) != 0) {
    fprintf(stderr, "#programs must be a power of two for fixed shuffle\n");
    return 1;
  }
  std::string interaction_pattern = GetFlag(FLAGS_interaction_pattern);
  if (params.fixed_shuffle && !interaction_pattern.empty()) {
    fprintf(stderr, "fixed shuffle and interaction pattern are incompatible\n");
    return 1;
  }

  constexpr size_t kColumnPadding1d = 4;
  size_t programs_per_column_1d = 0;
  size_t num_columns_1d = 0;
  std::string draw_to_1d = GetFlag(FLAGS_draw_to);
  if (!draw_to_1d.empty()) {
    programs_per_column_1d =
        params.num_programs /
        std::ceil(std::sqrt(params.num_programs /
                            (kSingleTapeSize + kColumnPadding1d)));
    num_columns_1d = (params.num_programs + programs_per_column_1d - 1) /
                     programs_per_column_1d;
  }
  std::vector<uint8_t> draw_buf_1d(
      programs_per_column_1d * 3 *
      (num_columns_1d * (kSingleTapeSize + kColumnPadding1d) -
       kColumnPadding1d));

  size_t grid_width_2d = 0;
  std::string draw_to_2d = GetFlag(FLAGS_draw_to_2d);
  if (!draw_to_2d.empty()) {
    if (GetFlag(FLAGS_grid_width_2d)) {
      grid_width_2d = GetFlag(FLAGS_grid_width_2d);
      if (params.num_programs % grid_width_2d != 0) {
        fprintf(stderr, "grid width must divide num_programs\n");
        return 1;
      }
    } else {
      grid_width_2d = std::sqrt(params.num_programs);
      if (grid_width_2d * grid_width_2d != params.num_programs) {
        fprintf(stderr, "number of programs must be a square\n");
        return 1;
      }
    }
  }
  static_assert(kSingleTapeSize == 64, "fix drawing if tapes are not 64 bytes");
  std::vector<uint8_t> draw_buf_2d(params.num_programs * 3 * kSingleTapeSize);

  if (!interaction_pattern.empty()) {
    FILE *f = fopen(interaction_pattern.c_str(), "r");
    if (!f) {
      fprintf(stderr, "could not open interaction pattern file\n");
      return 1;
    }
    params.allowed_interactions.resize(params.num_programs);
    long long a, b;
    while (fscanf(f, "%lld%lld", &a, &b) == 2) {
      if (static_cast<size_t>(a) >= params.num_programs ||
          static_cast<size_t>(b) >= params.num_programs) {
        fprintf(stderr,
                "invalid interaction pattern: programs not in [0, n) range.\n");
        return 1;
      }
      params.allowed_interactions[a].push_back(b);
    }
    fclose(f);
  }

  std::optional<std::string> log_to = GetFlag(FLAGS_log);
  uint32_t print_interval = GetFlag(FLAGS_print_interval);
  uint32_t clear_interval = GetFlag(FLAGS_clear_interval);
  std::optional<size_t> max_epochs = GetFlag(FLAGS_max_epochs);
  std::optional<size_t> stopping_bpb = GetFlag(FLAGS_stopping_bpb);

  if (params.save_interval % print_interval != 0) {
    fprintf(stderr, "save interval must be divisible by print interval\n");
    return 1;
  }

  if (clear_interval % print_interval != 0) {
    fprintf(stderr, "clear interval must be divisible by print interval\n");
    return 1;
  }

  params.callback_interval = print_interval;

  auto run_flag = GetFlag(FLAGS_run);
  auto lang = GetFlag(FLAGS_lang);
  const LanguageInterface *language = GetLanguage(lang);
  if (run_flag.has_value()) {
    printf("%s", ResetColors());
    language->RunSingleProgram(run_flag.value(), GetFlag(FLAGS_run_steps),
                               debug);
  } else {
    FILE *logfile = nullptr;
    if (log_to.has_value()) {
      logfile = CheckFopen(log_to->c_str(), "w");
      fprintf(logfile, "epoch,brotli_size,soup_size,higher_entropy\n");
    }

    auto callback = [&](const SimulationState &state) {
      if (!GetFlag(FLAGS_disable_output)) {
        if (state.epoch % clear_interval == 1) {
          printf("%s\033[2J\033[H", ResetColors());
        }
        printf(
            "%s\033[0;0H    Elapsed: %10.3f        ops: %23zu     "
            "MOps/s: %12.3f Epochs: %13zu ops/prog/epoch: %10.3f\n"
            "Brotli size: %10zu Brotli bpb: %23.4f bytes/prog: %12.4f     H0: "
            "%13.4f higher entropy: %10.6f\n",
            ResetColors(), state.elapsed_s, state.total_ops, state.mops_s,
            state.epoch, state.ops_per_run, state.brotli_size, state.brotli_bpb,
            state.bytes_per_prog, state.h0, state.higher_entropy);

        for (auto [s, f] : state.frequent_bytes) {
          printf("\033[37;1m%s%s %5.2f%% ", s.c_str(), ResetColors(),
                 f * 100.0);
        }
        printf("\n");
        for (auto [s, f] : state.uncommon_bytes) {
          printf("\033[37;1m%s%s %5.2f%% ", s.c_str(), ResetColors(),
                 f * 100.0);
        }
        printf("\n\n\n");

        for (size_t i = 0; i < std::min<size_t>(48, params.num_programs / 2);
             i++) {
          state.print_program(i);
        }
        fflush(stdout);
      }

      if (logfile) {
        fprintf(logfile, "%zu,%zu,%zu,%f\n", state.epoch, state.brotli_size,
                state.soup.size() / kSingleTapeSize, state.higher_entropy);
        fflush(logfile);
      }

      auto write_ppm = [](const std::string &base, size_t frame, size_t xs,
                          size_t ys, const std::vector<uint8_t> &data) {
        assert(data.size() == xs * ys * 3);
        std::string out_path(base.size() + 64, 0);
        out_path.resize(snprintf(out_path.data(), out_path.size(),
                                 "%s/%012lld.ppm", base.c_str(),
                                 (long long)frame));
        FILE *f = CheckFopen(out_path.c_str(), "w");
        fprintf(f, "P6\n%lld %lld\n255\n", (long long)xs, (long long)ys);
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
      };

      if (!draw_to_1d.empty()) {
        size_t xs = (kColumnPadding1d + kSingleTapeSize) * num_columns_1d -
                    kColumnPadding1d;
        for (size_t i = 0; i < params.num_programs; i++) {
          size_t column_id = i / programs_per_column_1d;
          size_t column_off = i % programs_per_column_1d;
          size_t x = column_id * (kColumnPadding1d + kSingleTapeSize);
          size_t y = column_off;
          for (size_t j = 0; j < kSingleTapeSize; j++) {
            memcpy(
                &draw_buf_1d[(y * xs + x + j) * 3],
                state.byte_colors[state.soup[i * kSingleTapeSize + j]].data(),
                3);
          }
        }
        write_ppm(draw_to_1d, state.epoch, xs, programs_per_column_1d,
                  draw_buf_1d);
      }

      if (!draw_to_2d.empty()) {
        static_assert(kSingleTapeSize == 64,
                      "fix drawing if tapes are not 64 bytes");
        size_t xs = grid_width_2d * 8;
        for (size_t i = 0; i < params.num_programs; i++) {
          size_t x = i % grid_width_2d;
          size_t y = i / grid_width_2d;
          for (size_t j = 0; j < kSingleTapeSize; j++) {
            size_t ix = j % 8;
            size_t iy = j / 8;
            size_t p = ((y * 8 + iy) * xs + x * 8 + ix) * 3;
            memcpy(
                &draw_buf_2d[p],
                state.byte_colors[state.soup[i * kSingleTapeSize + j]].data(),
                3);
            if (ix == 0 || iy == 0) {
              for (size_t c = 0; c < 3; ++c) {
                draw_buf_2d[p + c] = std::max(draw_buf_2d[p + c] - 32, 0);
              }
            }
          }
        }
        write_ppm(draw_to_2d, state.epoch, xs,
                  params.num_programs / grid_width_2d * 8, draw_buf_2d);
      }

      if (max_epochs.has_value() && state.epoch > *max_epochs) {
        return true;
      }
      if (stopping_bpb.has_value() && state.brotli_bpb < *stopping_bpb) {
        return true;
      };
      return false;
    };

    std::optional<std::string> initial_program = GetFlag(FLAGS_initial_program);
    language->RunSimulation(params, initial_program, callback);

    if (logfile) {
      fclose(logfile);
    }
  }
}
