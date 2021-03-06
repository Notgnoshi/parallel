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
NVXXFLAGS = -I$(INC_DIR) -std=c++11 #-gencode arch=compute_61,code=sm_61
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
	# The de/serialization tests require this directory to exist.
	mkdir -p $(BUILD_DIR)/tmp
	# Convert each of the text matrices to the expected binary format.
	find $(TEST_DIR)/matrices -name '*.txt' -type f -exec bash -c 'python3 pack.py "$$0" "$${0%.txt}.mat"' {} \;
	# Move the binary matrix files to the build directory.
	mv $(TEST_DIR)/matrices/*.mat $(BUILD_DIR)/tmp

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
	mkdir -p $(BUILD_DIR)/tmp
	python3 generate.py $(_SIZE) $(_SIZE) --output $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat
	python3 generate.py $(_SIZE) $(_SIZE) --output $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat
	python3 generate.py $(_SIZE) 1        --output $(BUILD_DIR)/tmp/$(_SIZE)x1_1.mat

_time_addition:
	@echo "============================="
	@echo "Timing Serial Addition"
	@echo "============================="
	$(BUILD_DIR)/$(TARGET) --time --kernel 1 --output $(BUILD_DIR)/tmp/time_addition_cpu.mat MMA $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat

	@echo "============================="
	@echo "Timing Parallel Addition"
	@echo "============================="
	$(BUILD_DIR)/$(TARGET) --time --kernel 2 --output $(BUILD_DIR)/tmp/time_addition_cuda.mat MMA $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat

_time_multiplication:
	@echo "============================="
	@echo "Timing Sequential M-V Multiplication"
	@echo "============================="
	$(BUILD_DIR)/$(TARGET) --time --kernel 1 --output $(BUILD_DIR)/tmp/time_multiplication_cpu.mat MVM $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x1_1.mat

	@echo "============================="
	@echo "Timing Parallel M-V Multiplication"
	@echo "============================="
	$(BUILD_DIR)/$(TARGET) --time --kernel 2 --output $(BUILD_DIR)/tmp/time_multiplication_cuda.mat MVM $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x1_1.mat

_time_extended_multiplication:
	@echo "============================="
	@echo "Timing Parallel M-M Multiplication"
	@echo "============================="
	$(BUILD_DIR)/$(TARGET) --time --kernel 2 --output $(BUILD_DIR)/tmp/time_ext_mult_cuda.mat MVM $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat

_addition:
	@echo "============================="
	@echo "Profiling M-M Addition:"
	@echo "============================="

	sed -i 's/BLOCK_SIZE( [0-9]\+, [0-9]\+, 1 )/BLOCK_SIZE( 1, 1, 1 )/' src/kernels/cuda_add.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "1x1 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MMA $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat --output $(BUILD_DIR)/tmp/addition_prof_result1.mat 2>&1 >/dev/null | grep AdditionKernel
	@echo "============================="

	sed -i 's/BLOCK_SIZE( [0-9]\+, [0-9]\+, 1 )/BLOCK_SIZE( 2, 2, 1 )/' src/kernels/cuda_add.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "2x2 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MMA $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat --output $(BUILD_DIR)/tmp/addition_prof_result2.mat 2>&1 >/dev/null | grep AdditionKernel
	@echo "============================="

	sed -i 's/BLOCK_SIZE( [0-9]\+, [0-9]\+, 1 )/BLOCK_SIZE( 4, 4, 1 )/' src/kernels/cuda_add.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "4x4 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MMA $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat --output $(BUILD_DIR)/tmp/addition_prof_result4.mat 2>&1 >/dev/null | grep AdditionKernel
	@echo "============================="

	sed -i 's/BLOCK_SIZE( [0-9]\+, [0-9]\+, 1 )/BLOCK_SIZE( 8, 8, 1 )/' src/kernels/cuda_add.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "8x8 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MMA $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat --output $(BUILD_DIR)/tmp/addition_prof_result8.mat 2>&1 >/dev/null | grep AdditionKernel
	@echo "============================="

	sed -i 's/BLOCK_SIZE( [0-9]\+, [0-9]\+, 1 )/BLOCK_SIZE( 16, 16, 1 )/' src/kernels/cuda_add.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "16x16 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MMA $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat --output $(BUILD_DIR)/tmp/addition_prof_result16.mat 2>&1 >/dev/null | grep AdditionKernel
	@echo "============================="

	sed -i 's/BLOCK_SIZE( [0-9]\+, [0-9]\+, 1 )/BLOCK_SIZE( 32, 32, 1 )/' src/kernels/cuda_add.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "32x32 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MMA $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat --output $(BUILD_DIR)/tmp/addition_prof_result32.mat 2>&1 >/dev/null | grep AdditionKernel
	@echo "============================="

	@echo "Reverting block size to 16x16"
	sed -i 's/BLOCK_SIZE( [0-9]\+, [0-9]\+, 1 )/BLOCK_SIZE( 16, 16, 1 )/' src/kernels/cuda_add.cu
	@echo
	@echo

_multiplication:
	@echo "============================="
	@echo "Profiling M-V Multiplication:"
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 1/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 1/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "1x1 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x1_1.mat --output $(BUILD_DIR)/tmp/multiplication_prof_result1.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 2/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 2/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "2x2 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x1_1.mat --output $(BUILD_DIR)/tmp/multiplication_prof_result2.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 4/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 4/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "4x4 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x1_1.mat --output $(BUILD_DIR)/tmp/multiplication_prof_result4.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 8/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 8/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "8x8 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x1_1.mat --output $(BUILD_DIR)/tmp/multiplication_prof_result8.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 16/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 16/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "16x16 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x1_1.mat --output $(BUILD_DIR)/tmp/multiplication_prof_result16.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 32/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 32/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "32x32 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x1_1.mat --output $(BUILD_DIR)/tmp/multiplication_prof_result32.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	@echo "Reverting block size to 16x16"
	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 16/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 16/' src/kernels/cuda_mult.cu
	@echo
	@echo

_extended_multiplication:
	@echo "============================="
	@echo "Profiling M-M Multiplication:"
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 1/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 1/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "1x1 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat --output $(BUILD_DIR)/tmp/extended_multiplication_prof_result1.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 2/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 2/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "2x2 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat --output $(BUILD_DIR)/tmp/extended_multiplication_prof_result2.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 4/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 4/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "4x4 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat --output $(BUILD_DIR)/tmp/extended_multiplication_prof_result4.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 8/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 8/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "8x8 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat --output $(BUILD_DIR)/tmp/extended_multiplication_prof_result8.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 16/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 16/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "16x16 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat --output $(BUILD_DIR)/tmp/extended_multiplication_prof_result16.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 32/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 32/' src/kernels/cuda_mult.cu
	@echo make
	@make all --no-print-directory -s -- 2>&1 /dev/null
	@echo "32x32 block size:"
	@echo "============================="
	nvprof $(BUILD_DIR)/$(TARGET) --kernel 2 MVM $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_1.mat $(BUILD_DIR)/tmp/$(_SIZE)x$(_SIZE)_2.mat --output $(BUILD_DIR)/tmp/extended_multiplication_prof_result32.mat 2>&1 >/dev/null | grep MultiplicationKernel
	@echo "============================="

	@echo "Reverting block size to 16x16"
	sed -i 's/BLOCK_XDIM [0-9]\+/BLOCK_XDIM 16/' src/kernels/cuda_mult.cu
	sed -i 's/BLOCK_YDIM [0-9]\+/BLOCK_YDIM 16/' src/kernels/cuda_mult.cu
	@echo
	@echo

# Avoid running/recompiling with debug symbols on some of the translation units.
profile: clean
profile: all
profile: _generate
profile: _addition
profile: _multiplication
profile: _extended_multiplication

time: clean
time: all
time: _generate
time: _time_addition
time: _time_multiplication
time: _time_extended_multiplication
