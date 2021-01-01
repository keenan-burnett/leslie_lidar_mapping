#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <boost/algorithm/string.hpp>
#include "pointmatcher/PointMatcher.h"

typedef PointMatcher<double> PM;
typedef PM::DataPoints DP;
using namespace PointMatcherSupport;  // NOLINT

/*!
   \brief Retrieves a vector of the (radar) file names in ascending order of time stamp
   \param datadir (absolute) path to the directory that contains (radar) files
   \param radar_files [out] A vector to be filled with a string for each file name
   \param extension Optional argument to specify the desired file extension. Files without this extension are rejected
*/
void get_file_names(std::string datadir, std::vector<std::string> &radar_files, std::string extension = "");

/*!
   \brief Retrieves lidar pointcloud data from an absolute file path.

   The ground truth transform associated with the pointcloud is associated with the timestamp of point [0].

   \param path absolute path to the lidar file
   \param pc 4 x N matrix of points (x, y, z, 1) x N
   \param intensties 1 x N matrix of infrared reflectivities / intensties associated with each point
   \param times vector of timestamps associated with each point. Seconds since the start of the hour in GPS time.

*/
void load_velodyne(std::string path, Eigen::MatrixXd &pc, Eigen::MatrixXd & intensities,
    std::vector<float> &times);

/*!
   \brief Retrieves an image from an absolute file path.
   \param path absolute path to the image file
   \param img The retrieved image will be placed here.
*/
void load_image(std::string path, cv::Mat &img);

/*!
   \brief Retrieve the ground truth pose and velocity information
   \param gtfile absolute path to the ground truth file (lidar_poses.csv)
   \param sensor_file Name of the sensor file for which we want ground truth (<timestamp>.bin)
   \param gt Output vector of ground truth information:
        GPSTime,x,y,z,vel_x,vel_y,vel_z,roll,pitch,heading,ang_vel_z
*/
bool get_groundtruth_data(std::string gtfile, std::string sensor_file, std::vector<double> &gt);

// Same as the function above except for odometry files with the following format:
// TIME1,TIME2,x,y,z,yaw,pitch,roll
// Output vector contents: x,y,z,yaw,pitch,roll
bool get_odom_data(std::string gtfile, std::string file1, std::string file2, std::vector<double> &gt);

Eigen::Matrix3d roll(double r);

Eigen::Matrix3d pitch(double p);

Eigen::Matrix3d yaw(double y);

/*!
   \brief Retrieve the 4x4 homogeneous transformation matrix from the line of groundtruth.
   \param gt Vector of ground truth information: GPSTime,x,y,z,vel_x,vel_y,vel_z,roll,pitch,heading,ang_vel_z
   \return 4x4 homogeneous transform: T_enu_sensor
*/
Eigen::Matrix4d getTransformFromGT(std::vector<double> gt);

/*!
   \brief Save a 4x4 homogeneous transform to a text file.
   \param path absolute path to the output text file.
   \param T input 4x4 homogeneous transform.
*/
void save_transform(std::string path, Eigen::Matrix4d T);

/*!
   \brief Load a 4x4 homogeneous transform from a text file.
   \param path absolute path to the input text file.
   \param T output 4x4 homogeneous transform.
*/
void load_transform(std::string path, Eigen::Matrix4d &T);

/*!
   \brief outputs a booleans depending on whether the frame is within a valid time window.
*/
bool filter_on_time(std::string fname, std::vector<std::vector<int>> valid_times);


/*!
   \brief Extracts the file name, ex: <timestamp> from <timestamp>.png
*/
void get_name_from_file(std::string file, std::string &name);


/*!
   \brief Prints the contents of a vector.
*/
template<typename T>
void print_vec(std::vector<T> v) {
    for (uint i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}

/*!
   \brief Calculates the BGR pixel value for each point that projects onto the image plane using bilinear interp
   \param pc Input pointcloud in lidar frame
   \param img Input image that corresponds to the lidar frame (closest in time)
   \param P 4x4 homogeneous camera perspective projection matrix [fx 0 cx 0; 0 fy cy 0; 0 0 1 0; 0 0 0 1]
   \param T_cam_lidar 4x4 homogeneous transformation matrix. transforms points from lidar frame to camera frame
   \param point_colors Output matrix (3 x N) of BGR pixel values. Default is zero for points without a color
*/
void get_point_colors(DP pc, cv::Mat img, Eigen::Matrix4d P, Eigen::MatrixXd T_cam_lidar,
    Eigen::MatrixXd &point_colors);

/*!
   \brief Where file names are UNIX EPOCH times, this function finds the closest_frame to the src_file within tgt_files.
*/
bool get_closest_frame(std::string src_file, std::vector<std::string> &tgt_files, std::string &closest_frame);

/*!
   \brief Colorizes a pointcloud by adding blue, green, red descriptors for all points that project onto the image plane
        The closest camera image in time is used for colorization. Default is zero for uncolored points.
   \param cloud Input pointcloud that will be colorized.
   \param T_enu_lidar 4x4 homogeneous transform from lidar frame to ENU frame
   \param P_cam 4x4 homogeneous camera perspective projection matrix [fx 0 cx 0; 0 fy cy 0; 0 0 1 0; 0 0 0 1]
   \param lidar_file Name of the current lidar file
   \param cam_files Vector of camera file names
   \param cam_pose_file path to the ground truth pose file for camera images (../../cam_poses.csv)
   \param root path to the root of the data directory
*/
void colorize_cloud(DP &cloud, Eigen::Matrix4d T_enu_lidar, Eigen::Matrix4d P_cam, std::string lidar_file,
    std::vector<std::string> &cam_files, std::string cam_pose_file, std::string root);

/*!
   \brief Load arguments from the command line and check their validity.
*/
int validateArgs(const int argc, const char *argv[], std::string &root, std::string &config);
