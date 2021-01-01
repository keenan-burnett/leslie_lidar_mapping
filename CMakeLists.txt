project(leslie)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wno-unused-parameter -O2")

add_definitions(-DSTEAM_DEFAULT_NUM_OPENMP_THREADS=8)

find_package(catkin REQUIRED COMPONENTS
	steam
)

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(OMP_NUM_THREADS 8)

find_package(OpenCV 3 REQUIRED)
find_package(libpointmatcher REQUIRED)
find_package(libnabo REQUIRED)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
    ${libpointmatcher_INCLUDE_DIRS}
	${libnabo_INCLUDE_DIRS}
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
  ${libnabo_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_mapping src/mapping.cpp)
target_link_libraries(${PROJECT_NAME}_mapping
  utils
  estimation
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${libpointmatcher_LIBRARIES}
)

add_executable(${PROJECT_NAME}_odometry src/odometry.cpp)
target_link_libraries(${PROJECT_NAME}_odometry
  utils
  estimation
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${libpointmatcher_LIBRARIES}
)

add_executable(batch_optimize_poses src/batch_optimize_poses.cpp)
target_link_libraries(batch_optimize_poses
	${DEPEND_LIBS}
	${catkin_LIBRARIES}
)
