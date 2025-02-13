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
#include <memory>
#include <unordered_map>

static std::unordered_map<std::string, std::unique_ptr<LanguageInterface>> *
registry() {
  static std::unordered_map<std::string, std::unique_ptr<LanguageInterface>> r;
  return &r;
}

const LanguageInterface *GetLanguage(const std::string &language) {
  auto map = registry();
  auto iter = map->find(language);
  if (iter == map->end()) {
    fprintf(stderr, "Unknown language `%s`\nAvailable languages:\n",
            language.c_str());
    for (const auto &[l, _] : *map) {
      fprintf(stderr, "%s\n", l.c_str());
    }
    exit(1);
  }
  return iter->second.get();
}

void RegisterLanguage(const char *lang,
                      std::unique_ptr<LanguageInterface> interface) {
  (*registry())[lang] = std::move(interface);
}
