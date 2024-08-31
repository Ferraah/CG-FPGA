# Compiler and flags
#CXX = icpx 
#CXXFLAGS = -O3 -Wall -Wextra -Iinclude -DINTEL_CL -g

# MPI Compiler and flags
MPICC = mpicxx
MPICXXFLAGS = -O3 -Wall -Wextra -Iinclude -DINTEL_CL 

AOCL_COMPILE_CONFIG := $(shell aocl compile-config)
AOCL_LINK_CONFIG := $(shell aocl link-config)

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Executable name
TARGET = cg_opencl 

# Default rule
all: $(TARGET)

# Rule to link the executable
$(TARGET): $(OBJS)
	$(MPICC) $(MPICXXFLAGS) $(AOCL_LINK_CONFIG) -o $@ $^

# Rule to compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(MPICC) $(MPICXXFLAGS) $(AOCL_COMPILE_CONFIG) -c $< -o $@

# Create the build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean up build files
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
