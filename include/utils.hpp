#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <boost/algorithm/string.hpp>

/*!
   \brief Retrieves a vector of the (radar) file names in ascending order of time stamp
   \param datadir (absolute) path to the directory that contains (radar) files
   \param radar_files [out] A vector to be filled with a string for each file name
   \param extension Optional argument to specify the desired file extension. Files without this extension are rejected
*/
void get_file_names(std::string datadir, std::vector<std::string> &radar_files, std::string extension = "");

void load_velodyne(std::string path, Eigen::MatrixXd &pc, Eigen::MatrixXd & intensities,
    std::vector<float> &times);

void load_image(std::string path, cv::Mat img);

bool get_groundtruth_data(std::string gtfile, std::string sensor_file, std::vector<double> &gt);

Eigen::Matrix4d getTransformFromGT(std::vector<double> gt);

Eigen::Matrix4d get_inverse_tf(Eigen::Matrix4d T);

void save_transform(std::string path, Eigen::Matrix4d T);

bool filter_on_time(std::string fname, std::vector<std::vector<int>> valid_times);

void get_name_from_file(std::string file, std::string &name);

template<typename T>
void print_vec(std::vector<T> v) {
    for (uint i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}
