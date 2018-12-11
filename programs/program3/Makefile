BUILD_DIR := build
SRC_DIR := src
INC_DIR := include
TEST_DIR := tests

TEST_TARGET := testsuite
TARGET := prog3

# Source files without the main entry point so I can link against the unit tests.
__SRC := $(shell find $(SRC_DIR) -name '*.cpp')
__OBJ := $(__SRC:%.cpp=$(BUILD_DIR)/%.o)

# Add the main entry point explicitly
SRC := $(__SRC) main.cpp
OBJ := $(SRC:%.cpp=$(BUILD_DIR)/%.o)

# Unit test source files
TEST_SRC := $(shell find $(TEST_DIR) -name '*.cpp')
TEST_OBJ := $(TEST_SRC:%.cpp=$(BUILD_DIR)/%.o)

DEP := $(OBJ:%.o=%.d) $(TEST_OBJ:%.o=%.d)

MPIXX := mpiCC
MPI := mpiexec
CXX := g++
LINK := g++

LINKFLAGS = -lm -fopenmp
# Get the right linker flags for use with gcc from the mpi compiler
LINKFLAGS += $(shell $(MPIXX) -showme:link)
CXXFLAGS = -I$(INC_DIR) -O3 -Wall -Wpedantic -Wextra -Werror -Wconversion -fopenmp -std=c++17 -x c++
# Get the right compile flags for use with gcc from the mpi compiler
CXXFLAGS += $(shell $(MPIXX) -showme:compile)

.PHONY: clean format docs viewdocs runtests

all: $(BUILD_DIR)/$(TARGET)

debug: clean
debug: CXXFLAGS += -g -O0
debug: all

testsuite: CXXFLAGS += -g -O0
testsuite: $(BUILD_DIR)/$(TEST_TARGET)

runtests: testsuite
	./$(BUILD_DIR)/$(TEST_TARGET)

# Avoid using $(OBJ) because it also contains main.o
$(BUILD_DIR)/$(TEST_TARGET): $(__OBJ) $(TEST_OBJ)
	mkdir -p $(@D)
	$(LINK) $^ -o $@ $(LINKFLAGS)

$(BUILD_DIR)/$(TARGET): $(OBJ)
	mkdir -p $(@D)
	$(LINK) $^ -o $@ $(LINKFLAGS)

# Apparently the location of this -include is what broke it before.
-include $(DEP)

$(BUILD_DIR)/%.o: %.cpp
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)/*

# Be sure to also include the main.cpp file, while not explicitly including .
format:
	find $(INC_DIR) $(SRC_DIR) $(TEST_DIR) ./main.cpp -name "*.cpp" -o -name "*.h" | xargs clang-format -style=file -i

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
	ssh perdition "mkdir -p $(BUILD_DIR); rm -rf $(BUILD_DIR)/*;"
	scp $(BUILD_DIR)/$(TARGET) $(BUILD_DIR)/$(TEST_TARGET) limbo:~/$(BUILD_DIR)
	scp $(BUILD_DIR)/$(TARGET) $(BUILD_DIR)/$(TEST_TARGET) perdition:~/$(BUILD_DIR)

cleanall: clean
cleanall:
	ssh limbo "rm -rf $(BUILD_DIR)/*"
	ssh perdition "rm -rf $(BUILD_DIR)/*"
