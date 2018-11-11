BUILD_DIR := build
SRC_DIR := src
INC_DIR := include
DOC_DIR := docs
TEST_DIR := tests

TARGET := prog2
TEST_TARGET := testsuite

_SIZE := 4000

SUBDIR_ROOTS = $(LIB_DIR) $(TEST_DIR)
CLEAN_DIRS := . $(shell find $(SUBDIR_ROOTS) -type d)
GARBAGE_PATTERNS := *.o *.d *.gch *.orig
GARBAGE := $(foreach DIR, $(CLEAN_DIRS), $(addprefix $(DIR)/, $(GARBAGE_PATTERNS)))

# Source files without the main entry point so I can link against the unit tests.
__SRC := $(shell find $(SRC_DIR) -name '*.cpp')
__OBJ := $(__SRC:%.cpp=$(BUILD_DIR)/%.o)

# Add the main entry point explicitly
SRC := $(__SRC) main.cpp
OBJ := $(SRC:%.cpp=$(BUILD_DIR)/%.o)

# CUDA source files
CU_SRC := $(shell find $(SRC_DIR) -name '*.cu')
CU_OBJ := $(CU_SRC:%.cu=$(BUILD_DIR)/%.o)

# Unit test source files
TEST_SRC := $(shell find $(TEST_DIR) -name '*.cpp')
TEST_OBJ := $(TEST_SRC:%.cpp=$(BUILD_DIR)/%.o)

DEP := $(OBJ:%.o=%.d) $(CU_OBJ:%.o=%.d) $(TEST_OBJ:%.o=%.d)

NVXX := nvcc
CXX := g++
LINK := g++

LINKFLAGS = -lm -lcuda -lcudart
NVXXFLAGS = -I$(INC_DIR) -std=c++11 -gencode arch=compute_61,code=sm_61
CXXFLAGS = -I$(INC_DIR) -O3 -Wall -Wpedantic -Wextra -Werror -Wconversion -std=c++11 -x c++

.PHONY: clean format docs viewdocs runtests

# TODO: This is supposed to notify make that header files are also dependencies,
#       but it doesn't work, and I don't know why.
# -include $(DEP)

all: $(BUILD_DIR)/$(TARGET)

debug: CXXFLAGS += -g -O0
debug: all

testsuite: CXXFLAGS += -g -O0
testsuite: NVXXFLAGS += -g -G -O0
testsuite: $(BUILD_DIR)/$(TEST_TARGET)

runtests: testsuite
	./$(BUILD_DIR)/$(TEST_TARGET)

# Avoid adding -lcppunit to the linker flags unless necessary. This way Dr. Karlsson can build the
# main executable without needing cppunit installed.
$(BUILD_DIR)/$(TEST_TARGET): LINKFLAGS += -lcppunit
# Avoid using $(OBJ) because it also contains main.o
$(BUILD_DIR)/$(TEST_TARGET): $(__OBJ) $(CU_OBJ) $(TEST_OBJ)
	mkdir -p $(@D)
	$(LINK) $^ -o $@ $(LINKFLAGS)

$(BUILD_DIR)/$(TARGET): $(OBJ) $(CU_OBJ)
	mkdir -p $(@D)
	$(LINK) $^ -o $@ $(LINKFLAGS)

$(BUILD_DIR)/%.o: %.cpp
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

# TODO: Figure out how to use --generate-dependencies in a way that doesn't
#       spam the console with system headers.
$(BUILD_DIR)/%.o: %.cu
	mkdir -p $(@D)
	$(NVXX) $(NVXXFLAGS) --compile $< -o $@

clean:
	rm -rf $(GARBAGE) $(BUILD_DIR)/*

# Be sure to also include the main.cpp file, while not explicitly including .
format:
	find $(INC_DIR) $(SRC_DIR) $(TEST_DIR) ./main.cpp -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" | xargs clang-format -style=file -i

# Avoid linting the unit tests because it throws a fit about cppunit.
lint:
	clang-tidy $(shell find $(INC_DIR) $(SRC_DIR) -name "*.cpp" -o -name "*.h") main.cpp -- $(CXXFLAGS)

docs:
	doxygen .doxyfile

viewdocs: docs
	firefox $(BUILD_DIR)/html/index.html &

_generate:
	python3 generate.py $(_SIZE) $(_SIZE) --output /tmp/$(_SIZE)x$(_SIZE)_1.mat
	python3 generate.py $(_SIZE) $(_SIZE) --output /tmp/$(_SIZE)x$(_SIZE)_2.mat
	python3 generate.py $(_SIZE) 1        --output /tmp/$(_SIZE)x1_1.mat

_addition:
	@echo "============================="
	@echo "Profiling M-M Addition:"
	@echo "============================="

	sed -i 's/BLOCK_SIZE( [0-9]\+, [0-9]\+, 1 )/BLOCK_SIZE( 4, 4, 1 )/' src/kernels/cuda_add.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "4x4 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MMA /tmp/$(_SIZE)x$(_SIZE)_1.mat /tmp/$(_SIZE)x$(_SIZE)_2.mat --output /tmp/addition_prof_result.mat 2>&1 >/dev/null | grep AdditionKernel
	@echo "============================="

	sed -i 's/BLOCK_SIZE( [0-9]\+, [0-9]\+, 1 )/BLOCK_SIZE( 8, 8, 1 )/' src/kernels/cuda_add.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "8x8 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MMA /tmp/$(_SIZE)x$(_SIZE)_1.mat /tmp/$(_SIZE)x$(_SIZE)_2.mat --output /tmp/addition_prof_result.mat 2>&1 >/dev/null | grep AdditionKernel
	@echo "============================="

	sed -i 's/BLOCK_SIZE( [0-9]\+, [0-9]\+, 1 )/BLOCK_SIZE( 16, 16, 1 )/' src/kernels/cuda_add.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "16x16 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MMA /tmp/$(_SIZE)x$(_SIZE)_1.mat /tmp/$(_SIZE)x$(_SIZE)_2.mat --output /tmp/addition_prof_result.mat 2>&1 >/dev/null | grep AdditionKernel
	@echo "============================="

	sed -i 's/BLOCK_SIZE( [0-9]\+, [0-9]\+, 1 )/BLOCK_SIZE( 32, 32, 1 )/' src/kernels/cuda_add.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "32x32 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MMA /tmp/$(_SIZE)x$(_SIZE)_1.mat /tmp/$(_SIZE)x$(_SIZE)_2.mat --output /tmp/addition_prof_result.mat 2>&1 >/dev/null | grep AdditionKernel
	@echo "============================="

	@echo "Reverting block size to 16x16"
	sed -i 's/BLOCK_SIZE( [0-9]\+, [0-9]\+, 1 )/BLOCK_SIZE( 16, 16, 1 )/' src/kernels/cuda_add.cu
	@echo
	@echo

_multiplication:
	@echo "============================="
	@echo "Profiling M-V Multiplication:"
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 4/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 4/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "4x4 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM /tmp/$(_SIZE)x$(_SIZE)_1.mat /tmp/$(_SIZE)x1_1.mat --output /tmp/multiplication_prof_result.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 8/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 8/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "8x8 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM /tmp/$(_SIZE)x$(_SIZE)_1.mat /tmp/$(_SIZE)x1_1.mat --output /tmp/multiplication_prof_result.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 16/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 16/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "16x16 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM /tmp/$(_SIZE)x$(_SIZE)_1.mat /tmp/$(_SIZE)x1_1.mat --output /tmp/multiplication_prof_result.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 32/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 32/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "32x32 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM /tmp/$(_SIZE)x$(_SIZE)_1.mat /tmp/$(_SIZE)x1_1.mat --output /tmp/multiplication_prof_result.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	@echo "Reverting block size to 16x16"
	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 16/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 16/' src/kernels/cuda_mult.cu
	@echo
	@echo

profile: all
profile: _generate
profile: _addition
profile: _multiplication
