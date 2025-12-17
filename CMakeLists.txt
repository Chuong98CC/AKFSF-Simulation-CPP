cmake_minimum_required(VERSION 3.10)
project(AKFSFSimulation)

set(CMAKE_CXX_STANDARD 11)

# Enable debug symbols for debugging
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

find_package (Eigen3 REQUIRED NO_MODULE)
find_package (OpenCV REQUIRED)
find_package (SDL2 REQUIRED)

# Find SDL2_ttf
find_path(SDL2_TTF_INCLUDE_DIR SDL_ttf.h PATH_SUFFIXES SDL2)
find_library(SDL2_TTF_LIBRARY SDL2_ttf)

set(SRC_CPP
    src/main.cpp
    src/display.cpp
    src/beacons.cpp
    src/sensors.cpp
    src/simulation.cpp
    src/utils.cpp
    src/kalmanfilter.cpp
)

add_executable(${PROJECT_NAME}
        ${SRC_CPP}
        )

target_include_directories(${PROJECT_NAME} PRIVATE 
        ${SDL2_INCLUDE_DIRS}
        ${SDL2_TTF_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}
        ${SDL2_LIBRARIES}
        ${SDL2_TTF_LIBRARY}
        Eigen3::Eigen
        ${OpenCV_LIBS}
        )

add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/data/ $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    
    