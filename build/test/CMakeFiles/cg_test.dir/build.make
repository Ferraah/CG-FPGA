# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /mnt/tier2/apps/USE/easybuild/staging/2023.1/software/CMake/3.26.3-GCCcore-12.3.0/bin/cmake

# The command to remove a file.
RM = /mnt/tier2/apps/USE/easybuild/staging/2023.1/software/CMake/3.26.3-GCCcore-12.3.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/users/u101373/CG-FPGA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/users/u101373/CG-FPGA/build

# Include any dependencies generated for this target.
include test/CMakeFiles/cg_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/cg_test.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/cg_test.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/cg_test.dir/flags.make

test/CMakeFiles/cg_test.dir/cg_test.cpp.o: test/CMakeFiles/cg_test.dir/flags.make
test/CMakeFiles/cg_test.dir/cg_test.cpp.o: /home/users/u101373/CG-FPGA/test/cg_test.cpp
test/CMakeFiles/cg_test.dir/cg_test.cpp.o: test/CMakeFiles/cg_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/users/u101373/CG-FPGA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/cg_test.dir/cg_test.cpp.o"
	cd /home/users/u101373/CG-FPGA/build/test && icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/cg_test.dir/cg_test.cpp.o -MF CMakeFiles/cg_test.dir/cg_test.cpp.o.d -o CMakeFiles/cg_test.dir/cg_test.cpp.o -c /home/users/u101373/CG-FPGA/test/cg_test.cpp

test/CMakeFiles/cg_test.dir/cg_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cg_test.dir/cg_test.cpp.i"
	cd /home/users/u101373/CG-FPGA/build/test && icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/users/u101373/CG-FPGA/test/cg_test.cpp > CMakeFiles/cg_test.dir/cg_test.cpp.i

test/CMakeFiles/cg_test.dir/cg_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cg_test.dir/cg_test.cpp.s"
	cd /home/users/u101373/CG-FPGA/build/test && icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/users/u101373/CG-FPGA/test/cg_test.cpp -o CMakeFiles/cg_test.dir/cg_test.cpp.s

# Object files for target cg_test
cg_test_OBJECTS = \
"CMakeFiles/cg_test.dir/cg_test.cpp.o"

# External object files for target cg_test
cg_test_EXTERNAL_OBJECTS =

test/cg_test: test/CMakeFiles/cg_test.dir/cg_test.cpp.o
test/cg_test: test/CMakeFiles/cg_test.dir/build.make
test/cg_test: lib/libgtest.a
test/cg_test: lib/libgtest_main.a
test/cg_test: lib/libgtest.a
test/cg_test: test/CMakeFiles/cg_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/users/u101373/CG-FPGA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cg_test"
	cd /home/users/u101373/CG-FPGA/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cg_test.dir/link.txt --verbose=$(VERBOSE)
	cd /home/users/u101373/CG-FPGA/build/test && /mnt/tier2/apps/USE/easybuild/staging/2023.1/software/CMake/3.26.3-GCCcore-12.3.0/bin/cmake -D TEST_TARGET=cg_test -D TEST_EXECUTABLE=/home/users/u101373/CG-FPGA/build/test/cg_test -D TEST_EXECUTOR= -D TEST_WORKING_DIR=/home/users/u101373/CG-FPGA/build/test -D TEST_EXTRA_ARGS= -D TEST_PROPERTIES= -D TEST_PREFIX= -D TEST_SUFFIX= -D TEST_FILTER= -D NO_PRETTY_TYPES=FALSE -D NO_PRETTY_VALUES=FALSE -D TEST_LIST=cg_test_TESTS -D CTEST_FILE=/home/users/u101373/CG-FPGA/build/test/cg_test[1]_tests.cmake -D TEST_DISCOVERY_TIMEOUT=120 -D TEST_XML_OUTPUT_DIR= -P /mnt/tier2/apps/USE/easybuild/staging/2023.1/software/CMake/3.26.3-GCCcore-12.3.0/share/cmake-3.26/Modules/GoogleTestAddTests.cmake

# Rule to build all files generated by this target.
test/CMakeFiles/cg_test.dir/build: test/cg_test
.PHONY : test/CMakeFiles/cg_test.dir/build

test/CMakeFiles/cg_test.dir/clean:
	cd /home/users/u101373/CG-FPGA/build/test && $(CMAKE_COMMAND) -P CMakeFiles/cg_test.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/cg_test.dir/clean

test/CMakeFiles/cg_test.dir/depend:
	cd /home/users/u101373/CG-FPGA/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/users/u101373/CG-FPGA /home/users/u101373/CG-FPGA/test /home/users/u101373/CG-FPGA/build /home/users/u101373/CG-FPGA/build/test /home/users/u101373/CG-FPGA/build/test/CMakeFiles/cg_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/cg_test.dir/depend

