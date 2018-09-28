INC_DIR := include
SRC_DIR := src
BUILD_DIR := build
DOC_DIR := docs

CIRCUIT_TARGET := circuit
PRIME_TARGET := prime

SUBDIR_ROOTS = $(INC_DIR) $(SRC_DIR) .
CLEAN_DIRS := . $(shell find $(SUBDIR_ROOTS) -type d)
GARBAGE_PATTERNS := *.o *.d *.gch *.orig
GARBAGE := $(foreach DIR, $(CLEAN_DIRS), $(addprefix $(DIR)/, $(GARBAGE_PATTERNS)))

# Maybe putting two programs in the same src tree was a mistake...
SRC := $(shell find $(SRC_DIR) -name '*.c')

# Apparently, if you name a source file the same as a target, it tries both the
# target and the default C target as if there was no Makefile...
CIRCUIT_SRC := $(SRC) cmain.c
PRIME_SRC := $(SRC) pmain.c

CIRCUIT_OBJ := $(CIRCUIT_SRC:%.c=$(BUILD_DIR)/%.o)
PRIME_OBJ := $(PRIME_SRC:%.c=$(BUILD_DIR)/%.o)

DEP := $(CIRCUIT_OBJ:%.o=%.d) $(PRIME_OBJ:%.o=%.d)

CXX := gcc
LINK := gcc
LINKFLAGS = -lm -fopenmp
CFLAGS = -I$(INC_DIR) -O3 -Wall -fopenmp -Wpedantic -Wextra -Werror -Wconversion -std=c11 -DTIMING

.PHONY: clean format docs viewdocs

-include $(DEP)

all: circuit prime

circuit: $(BUILD_DIR)/$(CIRCUIT_TARGET)
prime: $(BUILD_DIR)/$(PRIME_TARGET)

debug: CFLAGS += -g -O0
debug: all

$(BUILD_DIR)/$(CIRCUIT_TARGET): $(CIRCUIT_OBJ)
	mkdir -p $(@D)
	$(LINK) $^ -o $@ $(LINKFLAGS)

$(BUILD_DIR)/$(PRIME_TARGET): $(PRIME_OBJ)
	mkdir -p $(@D)
	$(LINK) $^ -o $@ $(LINKFLAGS)

$(BUILD_DIR)/%.o: %.c
	mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MMD -c $< -o $@

clean:
	rm -rf $(GARBAGE) $(BUILD_DIR)/* $(DOC_DIR)/html/

format:
	astyle --options=astyle.config --recursive "*.c" "*.h"

docs:
	doxygen $(DOC_DIR)/doxyfile

viewdocs: docs
	xdg-open $(DOC_DIR)/html/index.html
