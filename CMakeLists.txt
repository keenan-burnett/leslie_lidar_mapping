project(leslie)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wno-unused-parameter -O2")

find_package(OpenCV 3 REQUIRED)
find_package(libpointmatcher REQUIRED)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
    ${libpointmatcher_INCLUDE_DIRS}
)

add_library(utils src/utils.cpp)
target_link_libraries(utils
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${libpointmatcher_INCLUDE_DIRS}
)

add_library(estimation src/estimation.cpp)
target_link_libraries(estimation
  ${catkin_LIBRARIES}
)

add_executable(${PROJECT_NAME}_mapping src/mapping.cpp)
target_link_libraries(${PROJECT_NAME}_mapping
  utils
  estimation
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${libpointmatcher_LIBRARIES}
)
