BUILD_DIR := build
SRC_DIR := src
INC_DIR := include
DOC_DIR := docs
TEST_DIR := tests

TEST_TARGET := testsuite
TARGET := prog3

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
MPIXX := mpiCC
MPI := mpiexec
CXX := g++
LINK := g++

LINKFLAGS = -lm -lcuda -lcudart -fopenmp
# Get the right linker flags for use with gcc from the mpi compiler
LINKFLAGS += $(shell $(MPIXX) -showme:link)
NVXXFLAGS = -I$(INC_DIR) -O3 -std=c++11
CXXFLAGS = -I$(INC_DIR) -O3 -Wall -Wpedantic -Wextra -Werror -Wconversion -fopenmp -std=c++11 -x c++
# Get the right compile flags for use with gcc from the mpi compiler
CXXFLAGS += $(shell $(MPIXX) -showme:compile)

.PHONY: clean format docs viewdocs runtests

all: $(BUILD_DIR)/$(TARGET)

debug: clean
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

# Apparently the location of this -include is what broke it before.
-include $(DEP)

$(BUILD_DIR)/%.o: %.cpp
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

# TODO: Figure out how to use --generate-dependencies in a way that doesn't
#       spam the console with system headers.
$(BUILD_DIR)/%.o: %.cu
	mkdir -p $(@D)
	$(NVXX) $(NVXXFLAGS) --compile $< -o $@

clean:
	rm -rf $(BUILD_DIR)/*

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

# For use with my personal network so I don't need to use the Opp Lab.
deploy:
	@echo "Deploying to remote nodes"
	ssh limbo "mkdir -p $(BUILD_DIR); rm -rf $(BUILD_DIR)/*;"
	scp $(BUILD_DIR)/$(TARGET) $(BUILD_DIR)/$(TEST_TARGET) limbo:~/$(BUILD_DIR)

cleanall: clean
cleanall:
	ssh limbo "rm -rf $(BUILD_DIR)/*"

run:
	$(MPI) -np 24 $(BUILD_DIR)/$(TARGET) 4
