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
    std::string lidar_pose_file = root + "applanix/lidar_poses_ypr_ref2.csv";
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

    std::shared_ptr<PM::Transformation> rigidTrans;
    rigidTrans = PM::get().REG(Transformation).create("RigidTransformation");

    std::shared_ptr<PM::DataPointsFilter> removeScanner =
        PM::get().DataPointsFilterRegistrar.create("MinDistDataPointsFilter",
        {{"minDist", "2.0"}});

    std::shared_ptr<PM::DataPointsFilter> randSubsample =
        PM::get().DataPointsFilterRegistrar.create("RandomSamplingDataPointsFilter",
        {{"prob", toParam(0.90)}});

    std::deque<DP> sliding_window;
    uint window_size = 10;
    Eigen::Matrix4d T_enu_map = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_prev = Eigen::Matrix4d::Identity();

    uint start = 92;
    for (uint i = start; i < lidar_files.size(); ++i) {
        std::cout << i << " / " << lidar_files.size() - 1 << std::endl;
        Eigen::MatrixXd pc, intensities;
        std::vector<float> times;
        load_velodyne(root + "lidar/" + lidar_files[i], pc, intensities, times);
        std::vector<double> gt;
        // GPSTime,x,y,z,vel_x,vel_y,vel_z,roll,pitch,heading,ang_vel_z
        assert(get_groundtruth_data(lidar_pose_file, lidar_files[i], gt));
        Eigen::Matrix4d T_enu_sensor = getTransformFromGT(gt);
        removeMotionDistortion(pc, times, T_enu_sensor, gt);

        DP newCloud(pc, labels);
        newCloud = removeScanner->filter(newCloud);
        if (i == start) {
            T_enu_map = T_enu_sensor;
            sliding_window.push_back(newCloud);
            continue;
        }
        Eigen::Matrix4d T_map_sensor = get_inverse_tf(T_enu_map) * T_enu_sensor;
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        try {
            DP submap = sliding_window[0];
            for (uint j = 1; j < sliding_window.size(); ++j) {
                submap.concatenate(sliding_window[j]);
            }
            submap = randSubsample->filter(submap);  // random subsample to speed up ICP
            T = icp(randSubsample->filter(newCloud), submap, T_map_sensor);
        } catch (PM::ConvergenceError& error) {
            std::cout << "ERROR PM::ICP failed to converge: " << error.what() << std::endl;
            return 1;
        }
        std::cout << T << std::endl;
        DP transformed = rigidTrans->compute(newCloud, T);
        sliding_window.push_back(transformed);
        if (sliding_window.size() > window_size)
            sliding_window.pop_front();

        Eigen::Matrix4d T_rel = get_inverse_tf(T_prev) * T;
        T_prev = T;

        ofs.open(lidar_odom_file, std::ios::app);
        std::string time1, time2;
        get_name_from_file(lidar_files[i - 1], time1);
        get_name_from_file(lidar_files[i], time2);
        double yaw = 0, pitch = 0, roll = 0;
        Eigen::Matrix3d C = T_rel.block(0, 0, 3, 3);
        rotToYawPitchRoll(C, yaw, pitch, roll);
        ofs << time1 << "," << time2 << "," << T_rel(0, 3) << "," << T_rel(1, 3) << "," << T_rel(2, 3) << ","
            << yaw << "," << pitch << "," << roll << "\n";
        ofs.close();
    }
}
