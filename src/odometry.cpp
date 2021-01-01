#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <boost/algorithm/string.hpp>
#include "estimation.hpp"
#include "utils.hpp"
#include "pointmatcher/PointMatcher.h"

typedef PointMatcher<double> PM;
typedef PM::DataPoints DP;
using namespace PointMatcherSupport;  // NOLINT

int main(int argc, const char *argv[]) {
    std::string root, config;
    if (validateArgs(argc, argv, root, config) != 0) {
        return 1;
    }
    std::vector<std::string> lidar_files;
    get_file_names(root + "lidar/", lidar_files, "bin");
    std::string lidar_pose_file = root + "applanix/lidar_poses.csv";
    std::string lidar_odom_file = root + "applanix/lidar_odom_poses.csv";
    std::ofstream ofs;
    ofs.open(lidar_odom_file, std::ios::out);
    ofs << "TIME1,TIME2,x,y,z,yaw,pitch,roll\n";
    ofs.close();

    PM::ICP icp;
    if (config.empty()) {
        icp.setDefault();
    } else {
        std::ifstream ifs(config.c_str());
        icp.loadFromYaml(ifs);
    }

    DP::Labels labels;
    labels.push_back(DP::Label("x", 1));
    labels.push_back(DP::Label("y", 1));
    labels.push_back(DP::Label("z", 1));
    labels.push_back(DP::Label("w", 1));

    std::shared_ptr<PM::DataPointsFilter> removeScanner =
        PM::get().DataPointsFilterRegistrar.create("MinDistDataPointsFilter",
        {{"minDist", "2.0"}});

    DP cloud1, cloud2;

    for (uint i = 0; i < lidar_files.size(); ++i) {
        std::cout << i << " / " << lidar_files.size() - 1 << std::endl;
        Eigen::MatrixXd pc, intensities;
        std::vector<float> times;
        load_velodyne(root + "lidar/" + lidar_files[i], pc, intensities, times);
        std::vector<double> gt;
        // GPSTime,x,y,z,vel_x,vel_y,vel_z,roll,pitch,heading,ang_vel_z
        assert(get_groundtruth_data(lidar_pose_file, lidar_files[i], gt));
        Eigen::Matrix4d T_enu_sensor = getTransformFromGT(gt);
        removeMotionDistortion(pc, times, T_enu_sensor, gt);

        cloud1 = cloud2;
        cloud2 = DP(pc, labels);
        cloud2 = removeScanner->filter(cloud2);
        if (i == 0)
            continue;
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        try {
            T = icp(cloud2, cloud1);
        } catch (PM::ConvergenceError& error) {
            std::cout << "ERROR PM::ICP failed to converge: " << error.what() << std::endl;
            return 1;
        }
        ofs.open(lidar_odom_file, std::ios::app);
        std::string time1, time2;
        get_name_from_file(lidar_files[i - 1], time1);
        get_name_from_file(lidar_files[i], time2);
        double yaw = 0, pitch = 0, roll = 0;
        Eigen::Matrix3d C = T.block(0, 0, 3, 3);
        rotToYawPitchRoll(C, yaw, pitch, roll);
        ofs << time1 << "," << time2 << "," << T(0, 3) << "," << T(1, 3) << "," << T(2, 3) << ","
            << yaw << "," << pitch << "," << roll << "\n";
        ofs.close();
    }
}
