cmake_minimum_required(VERSION 2.8)

project(homography_global_optimization)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# homography_global_optimization
add_executable(homography_global_optimization homography_global_optimization.cpp)
target_link_libraries(homography_global_optimization ${CERES_LIBRARIES})
