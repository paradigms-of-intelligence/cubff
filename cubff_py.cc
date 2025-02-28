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

#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>);

PYBIND11_MODULE(cubff, m) {
  m.doc() = "cubff python module";

  pybind11::class_<SimulationParams>(m, "SimulationParams")
      .def_readwrite("num_programs", &SimulationParams::num_programs)
      .def_readwrite("seed", &SimulationParams::seed)
      .def_readwrite("mutation_prob", &SimulationParams::mutation_prob)
      .def_readwrite("reset_interval", &SimulationParams::reset_interval)
      .def_readwrite("load_from", &SimulationParams::load_from)
      .def_readwrite("save_to", &SimulationParams::save_to)
      .def_readwrite("callback_interval", &SimulationParams::callback_interval)
      .def_readwrite("save_interval", &SimulationParams::save_interval)
      .def_readwrite("permute_programs", &SimulationParams::permute_programs)
      .def_readwrite("fixed_shuffle", &SimulationParams::fixed_shuffle)
      .def_readwrite("zero_init", &SimulationParams::zero_init)
      .def_readwrite("allowed_interactions",
                     &SimulationParams::allowed_interactions)
      .def_readwrite("eval_selfrep", &SimulationParams::eval_selfrep)
      .def(pybind11::init<>());

  py::bind_vector<std::vector<uint8_t>>(m, "VectorUint8",
                                        py::buffer_protocol());
  py::bind_vector<std::vector<uint32_t>>(m, "VectorUint32",
                                         py::buffer_protocol());

  pybind11::class_<SimulationState>(m, "SimulationState")
      .def_readonly("soup", &SimulationState::soup)
      .def_readonly("shuffle_idx", &SimulationState::shuffle_idx)
      .def_readonly("elapsed_s", &SimulationState::elapsed_s)
      .def_readonly("total_ops", &SimulationState::total_ops)
      .def_readonly("mops_s", &SimulationState::mops_s)
      .def_readonly("epoch", &SimulationState::epoch)
      .def_readonly("ops_per_run", &SimulationState::ops_per_run)
      .def_readonly("brotli_size", &SimulationState::brotli_size)
      .def_readonly("brotli_bpb", &SimulationState::brotli_bpb)
      .def_readonly("bytes_per_prog", &SimulationState::bytes_per_prog)
      .def_readonly("h0", &SimulationState::h0)
      .def_readonly("higher_entropy", &SimulationState::higher_entropy)
      .def_readonly("frequent_bytes", &SimulationState::frequent_bytes)
      .def_readonly("uncommon_bytes", &SimulationState::uncommon_bytes)
      .def_readonly("replication_per_prog",
                    &SimulationState::replication_per_prog);

  pybind11::class_<LanguageInterface>(m, "LanguageInterface")
      .def("PrintProgram",
           [](const LanguageInterface* interface, size_t pc_pos,
              const std::vector<uint8_t>& program,
              const std::vector<size_t>& separators) {
             interface->PrintProgram(pc_pos, program.data(), program.size(),
                                     separators.data(), separators.size());
           })
      .def("RunSimulation", &LanguageInterface::RunSimulation)
      .def("RunSingleProgram", &LanguageInterface::RunSingleProgram)
      .def("RunSingleParsedProgram", &LanguageInterface::RunSingleParsedProgram)
      .def("EvalSelfrep", &LanguageInterface::EvalSelfrep)
      .def("EvalParsedSelfrep", &LanguageInterface::EvalParsedSelfrep)
      .def("Parse", &LanguageInterface::Parse);

  m.def("GetLanguage", &GetLanguage, py::return_value_policy::reference);
  m.def("ResetColors", []() {
    printf("%s", ResetColors());
    fflush(stdout);
  });
  m.attr("kSelfrepThreshold") = kSelfrepThreshold;
}
