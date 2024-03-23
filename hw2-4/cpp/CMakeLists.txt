cmake_minimum_required (VERSION 3.16) #Usually the latest one you are using
project(My_Torch_Project DESCRIPTION "My Torch Project")

# Ref: https://gist.github.com/rohit-kumar-j/c42f244f5f3d49c5fd39e15efc09a4a2

# --------- Optionally enable standards ---------
# ------------------ STANDARDS ------------------
enable_language(C)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(BUILD_MODE Debug)
SET(CMAKE_CXX_FLAGS "-Wall -g")
SET(CMAKE_CXX_FLAGS_DEBUG "-g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
enable_testing()     # --- This is placed specifically at the top of the file to enable testing

# ------------------ Path Settings ------------------
set(MAIN_PATH "${CMAKE_SOURCE_DIR}") # ------- Makes source dir
set(MY_MAIN_EXECUTABLE "hw")
set(CMAKE_PREFIX_PATH "${MAIN_PATH}/libtorch")
# set(ANOTHER_DEPENDENCY_PATH "${MAIN_PATH}/external/another_dependency")
set(SRC "${MAIN_PATH}/src")

# ------------------ Torch Settings ------------------
include_directories("${MAIN_PATH}/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# -------------------- Add Tests Subdirectory --------------------
# add_subdirectory(${SRC}/test)

# ------------------ Main Executable ------------------
add_executable ("${MY_MAIN_EXECUTABLE}" "${SRC}/hw.cpp")
target_link_libraries("${MY_MAIN_EXECUTABLE}" "${TORCH_LIBRARIES}")

# -------------------- REMOVE WINDOWS ERRORS --------------------
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET "${MY_MAIN_EXECUTABLE}" 
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:"${MY_MAIN_EXECUTABLE}" >)
endif (MSVC)