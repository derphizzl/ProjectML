cmake_minimum_required(VERSION 3.16)
set(ProjectName ProjectML)
project(${ProjectName} CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable OpenMP
find_package(OpenMP REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

list(APPEND SOURCES
    main.cpp
)

include(${CMAKE_CURRENT_LIST_DIR}/Logging/Logging.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Primitive/Primitive.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Algorithms/Algorithms.cmake)

# Source files
add_executable(${ProjectName}
    ${SOURCES}
    ${HEADERS}
)

#Include directories
target_include_directories(${ProjectName} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${INCPATHS}
)

# Link OpenCV and OpenMP
target_link_libraries(${ProjectName}
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

target_compile_definitions(${ProjectName} PRIVATE
    ${DEFINES}
)
