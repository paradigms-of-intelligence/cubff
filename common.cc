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

#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <unordered_map>

template <typename T>
std::unordered_map<std::string, T> *reg() {
  static std::unordered_map<std::string, T> r;
  return &r;
}

template <typename T>
static T GetFn(const std::string &language) {
  auto map = reg<T>();
  auto iter = map->find(language);
  if (iter == map->end()) {
    fprintf(stderr, "Unknown language `%s`\nAvailable languages:\n",
            language.c_str());
    for (auto [l, _] : *map) {
      fprintf(stderr, "%s\n", l.c_str());
    }
    exit(1);
  }
  return iter->second;
}

void RunSingleProgram(const std::string &language, std::string program,
                      size_t stepcount, bool debug) {
  GetFn<runsingle_t>(language)(program, stepcount, debug);
}

void RunSimulation(const std::string &language, const SimulationParams &params,
                   std::optional<std::string> initial_program,
                   std::function<bool(const SimulationState &)> callback) {
  GetFn<runsimulation_t>(language)(params, initial_program, callback);
}

void RegisterLanguage(const char *lang, runsingle_t runsingle,
                      runsimulation_t runsimulation) {
  (*reg<runsingle_t>())[lang] = runsingle;
  (*reg<runsimulation_t>())[lang] = runsimulation;
}
