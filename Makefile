
MACOS := $(if $(filter Darwin,$(shell uname -s)),1)

ifeq (${MACOS}, 1)
	CUDA := 0
else
	CUDA := 1
endif

PYTHON := 0

COMMON_FLAGS := -g -std=c++17 -O3 \
								$(shell pkg-config --cflags libbrotlienc libbrotlicommon) \
								${EXTRA_LDFLAGS}

LINK_FLAGS := $(shell pkg-config --libs libbrotlienc libbrotlicommon)


ifeq (${CUDA}, 1)
CUDA_PATH := $(shell which nvcc | xargs realpath | xargs dirname | xargs dirname)
FLAGS := ${COMMON_FLAGS} -arch sm_75 --compiler-options -Wall,-fPIC \
					--compiler-bindir $(shell which ${CXX}) \
					-I ${CUDA_PATH}/include -L ${CUDA_PATH}/lib
COMPILE_FLAGS := ${FLAGS}
COMPILER := nvcc
else
	FLAGS := ${COMMON_FLAGS} -Wall -fPIC

	ifeq (${MACOS}, 1)
		FLAGS += -Xclang -fopenmp
		LINK_FLAGS += -lomp -undefined dynamic_lookup
	else
		FLAGS += -fopenmp
	endif

	COMPILE_FLAGS := ${FLAGS} -xc++
	COMPILER := ${CXX}
endif

LANGS=$(patsubst %.cu,build/%.o,$(wildcard *.cu))

PYEXT=$(shell python3-config --extension-suffix)

ifeq (${PYTHON}, 0)
.PHONY:
all: bin/main
else
all: bin/main bin/cubff${PYEXT}
endif

bin/main: build/main.o build/common.o ${LANGS}
	${COMPILER} $^ ${FLAGS} ${LINK_FLAGS} -o $@

build/%.o: %.cc common.h common_language.h
	${COMPILER} -c ${COMPILE_FLAGS} $< -o $@

build/%.o: %.cu common.h common_language.h forth.inc.h
	${COMPILER} -c ${COMPILE_FLAGS} $< -o $@

build/cubff_py.o: cubff_py.cc common.h
	${COMPILER} -c ${COMPILE_FLAGS} $< -o $@ $(shell python3 -m pybind11 --includes)

bin/cubff${PYEXT}: build/cubff_py.o build/common.o ${LANGS}
	${COMPILER} -shared $^ ${FLAGS} ${LINK_FLAGS} -o $@

.PHONY:
clean:
	rm -rf bin/main bin/cubff* build/*
