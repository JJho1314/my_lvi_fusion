cmake_minimum_required(VERSION 2.8.3)
project(lvi_fusion)


# set(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++17 -O3 -lboost_system  -pthread -Wenum-compare") # -Wall  -msse2 -msse3
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(-w)
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# Add OpenCV if available
# set(OpenCV_DIR "/home/zivlin/opt/opencv/build")
FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem iostreams program_options system serialization)
if(Boost_FOUND)
	INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
	LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
endif()

find_package(catkin REQUIRED COMPONENTS
                    roscpp
                    rospy
                    std_msgs
                    geometry_msgs
                    nav_msgs
                    tf
                    cv_bridge
                    message_generation
                    pcl_conversions
                    sensor_msgs
                    livox_ros_driver
                    
            )

find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV 3.4 REQUIRED)
find_package(GTSAM REQUIRED QUIET)
add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

### Find OpenMP #######
FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	#cmake only check for separate OpenMP library on AppleClang 7+
	#https://github.com/Kitware/CMake/blob/42212f7539040139ecec092547b7d58ef12a4d72/Modules/FindOpenMP.cmake#L252
	if (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang" AND (NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.0"))
		SET(OpenMP_LIBS ${OpenMP_libomp_LIBRARY})
		LIST(APPEND OpenMVS_EXTRA_LIBS ${OpenMP_LIBS})
	endif()
else()
	message("-- Can't find OpenMP. Continuing without it.")
endif()

# find_package(OpenMVS)
if(OpenMVS_FOUND)
	include_directories(${OpenMVS_INCLUDE_DIRS})
	add_definitions(${OpenMVS_DEFINITIONS})
endif()

generate_messages(
 DEPENDENCIES
 geometry_msgs
 std_msgs
 nav_msgs
 sensor_msgs
)

# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")
catkin_package(
  INCLUDE_DIRS include
  DEPENDS PCL GTSAM

  CATKIN_DEPENDS 
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

include_directories(${catkin_INCLUDE_DIRS})



set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${livox_ros_driver_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src/factor
  ./src
  ./src/loam/include
  ./src/tools/
  ./src/rgb_map 

)

link_directories(
	include
	${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
)


message(STATUS "===== ${PROJECT_NAME}: OpenCV library status: =====")
message(STATUS "===== OpenCV ersion: ${OpenCV_VERSION}  =====")
message(STATUS "===== OpenCV libraries: ${OpenCV_LIBS} =====")
message(STATUS "===== OpenCV include path: ${OpenCV_INCLUDE_DIRS} =====")

add_executable(lvi_fusion_LiDAR_front_end src/loam/LiDAR_front_end.cpp)
target_link_libraries(lvi_fusion_LiDAR_front_end ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
# if(Ceres_FOUND)
#   message(STATUS "===== Find ceres, Version ${Ceres_VERSION} =====")  
#   include_directories(${CERES_INCLUDE_DIRS})
#   add_executable(lvi_fusion_cam_cali src/lvi_fusion_cam_cali.cpp)
#   target_link_libraries(lvi_fusion_cam_cali ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})

#   add_executable(lvi_fusion_cam_cali_create_cali_board src/lvi_fusion_cam_cali_create_cali_board.cpp)
#   target_link_libraries(lvi_fusion_cam_cali_create_cali_board ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
# endif()
add_executable(test_timer src/tools/test_timer.cpp)


file(GLOB mySRC_LIST
src/factor/*.cpp
)

add_executable(lvi_fusion_mapping src/lvi_fusion.cpp 
                src/lvi_fusion_lio.cpp
                src/loam/include/kd_tree/ikd_Tree.cpp
                src/loam/include/FOV_Checker/FOV_Checker.cpp 
                src/loam/IMU_Processing.cpp
                src/rgb_map/offline_map_recorder.cpp
                # From VIO
                src/lvi_fusion_vio.cpp
                src/optical_flow/lkpyramid.cpp
                src/rgb_map/rgbmap_tracker.cpp
                src/rgb_map/image_frame.cpp
                src/rgb_map/pointcloud_rgbd.cpp
                ${mySRC_LIST}
              )
target_link_libraries(lvi_fusion_mapping 
                          ${catkin_LIBRARIES}
                          ${Boost_LIBRARIES}
                          ${Boost_FILESYSTEM_LIBRARY}
                          ${Boost_SERIALIZATION_LIBRARY} # serialization
                          ${OpenCV_LIBRARIES}
                          ${CERES_LIBRARIES} 
                          # ${OpenMVS_LIBRARIES}
                          pcl_common 
                          pcl_io) 

# add_executable(map_build src/map_build.cpp
# ${PROJECT_SOURCE_DIR}/include/Scancontext.cpp
#                 )
# add_dependencies(map_build ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
# target_compile_options(map_build PRIVATE ${OpenMP_CXX_FLAGS})
# target_link_libraries(map_build ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} ${CERES_LIBRARIES} ${Boost_LIBRARIES}
# ${Boost_FILESYSTEM_LIBRARY} ${Boost_SERIALIZATION_LIBRARY} gtsam)          

add_executable(map_build 
  src/map_build.cpp
  ${PROJECT_SOURCE_DIR}/include/Scancontext.cpp
)
target_compile_options(map_build 
  PRIVATE ${OpenMP_CXX_FLAGS}
)
target_link_libraries(map_build 
  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  gtsam
)


# Range Image Projection
# add_executable(${PROJECT_NAME}_imageProjection src/loam/ScanPreprocess.cpp)
# add_dependencies(${PROJECT_NAME}_imageProjection ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
# target_link_libraries(${PROJECT_NAME}_imageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS})

catkin_install_python(PROGRAMS src/read_image.py
  src/read_CompressedImage.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


