#include <string>
#include <vector>
#include <deque>
#include <iostream>
#include <fstream>
#include <chrono>
#include <boost/algorithm/string.hpp>
#include "estimation.hpp"
#include "utils.hpp"
#include "pointmatcher/PointMatcher.h"

// #include "pointmap/pointmap.h"
// #include "polar_processing/polar_processing.h"

typedef PointMatcher<double> PM;
typedef PM::DataPoints DP;
using namespace PointMatcherSupport;  // NOLINT

void getMapFrames(std::string root, std::vector<std::string> &frames) {
    frames.clear();
    std::ifstream ifs;
    ifs.open(root + "map/frames.txt", std::ios::in);
    std::string line;
    std::getline(ifs, line);  // clear out the csv file header before searching
    while (std::getline(ifs, line)) {
        std::vector<std::string> parts;
        boost::split(parts, line, boost::is_any_of("."));
        frames.push_back(root + "map/frames/" + parts[0] + ".ply");
    }
}

void getSubMap(std::string root, std::vector<int> closestK, DP &submap){
    std::vector<std::string> frames;
    getMapFrames(root, frames);
    submap = DP::load(frames[closestK[0]]);
    for (uint i = 1; i < closestK.size(); ++i) {
        submap.concatenate(DP::load(frames[closestK[i]]));
    }
}

void generateFinalMap(std::string root, DP &map) {
    std::shared_ptr<PM::DataPointsFilter> ocTreeSubsample =
        PM::get().DataPointsFilterRegistrar.create("OctreeGridDataPointsFilter",
        {{"maxSizeByNode", "0.063"}, {"samplingMethod", "1"}});
    std::vector<std::string> frames;
    getMapFrames(root, frames);
    map = DP::load(frames[0]);
    for (uint i = 1; i < frames.size(); ++i) {
        map.concatenate(DP::load(frames[i]));
    }
    map = ocTreeSubsample->filter(map);
}

int main() {
    // std::string root = "/media/keenan/autorontossd1/2020_11_05/";
    std::string root = "/home/keenan/Documents/data/boreas/2020_11_05/";
    std::vector<std::string> lidar_files;
    get_file_names(root + "lidar/", lidar_files, "bin");
    std::string lidar_pose_file = root + "applanix/lidar_poses.csv";
    std::string camera_pose_file = root + "applanix/camera_poses.csv";
    std::vector<std::vector<int>> valid_times{{1604603469, 1604603598}, {1604603692, 1604603857},
        {1604603957, 1604604168}, {1604604278, 1604604445}};
    std::ofstream ofs;
    ofs.open(root + "map/frames.txt", std::ios::out);
    ofs << "frame_name\n";
    ofs.close();

    PM::ICP icp;
    icp.setDefault();
    DP::Labels labels;
    labels.push_back(DP::Label("x", 1));
    labels.push_back(DP::Label("y", 1));
    labels.push_back(DP::Label("z", 1));
    labels.push_back(DP::Label("w", 1));
    DP::Labels desclabels;
    desclabels.push_back(DP::Label("intensity", 1));
    std::shared_ptr<PM::Transformation> rigidTrans;
    rigidTrans = PM::get().REG(Transformation).create("RigidTransformation");
    PM::TransformationParameters T = Eigen::Matrix4d::Identity();

    std::shared_ptr<PM::DataPointsFilter> removeScanner =
        PM::get().DataPointsFilterRegistrar.create("MinDistDataPointsFilter",
        {{"minDist", "1.5"}});

    // This filter will randomly remove 35% of the points.
    std::shared_ptr<PM::DataPointsFilter> randSubsample =
        PM::get().DataPointsFilterRegistrar.create("RandomSamplingDataPointsFilter",
        {{"prob", toParam(0.65)}});

    std::shared_ptr<PM::DataPointsFilter> densityFilter =
        PM::get().DataPointsFilterRegistrar.create("SurfaceNormalDataPointsFilter",
        {{"knn", "10"}, {"epsilon", "5"}, {"keepDensities", "1"}, {"keepNormals", "0"}});

    std::shared_ptr<PM::DataPointsFilter> uniformSubsample =
        PM::get().DataPointsFilterRegistrar.create(
        "MaxDensityDataPointsFilter",
        {{"maxDensity", toParam(4000)}});

    std::shared_ptr<PM::DataPointsFilter> ocTreeSubsample =
        PM::get().DataPointsFilterRegistrar.create("OctreeGridDataPointsFilter",
        {{"maxSizeByNode", "0.063"}, {"samplingMethod", "1"}});

    DP map;
    bool map_init = false;
    Eigen::Matrix4d T_enu_map = Eigen::Matrix4d::Identity();
    bool t_enu_map_init = false;
    double prev_x = 0, prev_y = 0;
    std::vector<std::vector<float>> frame_loc;
    uint retrieveK = 10;

    for (uint i = 0; i < lidar_files.size(); ++i) {
        std::cout << i << " / " << lidar_files.size() - 1;
        // Filter out frames outside of the times of interest
        if (!filter_on_time(lidar_files[i], valid_times)) {
            std::cout << " skipping..." << std::endl;
            continue;
        }
        // Get the ground truth data for this frame
        std::vector<double> gt;
        // GPSTime,x,y,z,vel_x,vel_y,vel_z,roll,pitch,heading,ang_vel_z
        assert(get_groundtruth_data(lidar_pose_file, lidar_files[i], gt));
        // Filter out frames when we haven't moved much since the last frame
        double d = sqrt(pow(prev_x - gt[1], 2) + pow(prev_y - gt[2], 2));
        if (d < 1.0) {
            std::cout << " skipping..." << std::endl;
            continue;
        }

        // This frame is being used, so add it to the list
        std::ofstream ofs;
        ofs.open(root + "map/frames.txt", std::ios::app);
        ofs << lidar_files[i] << "\n";
        ofs.close();
        std::cout << " processing..." << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();

        // Load in points
        Eigen::MatrixXd pc;
        Eigen::MatrixXd intensities;
        std::vector<float> times;
        load_velodyne(root + "lidar/" + lidar_files[i], pc, intensities, times);

        Eigen::Matrix4d T_enu_sensor = getTransformFromGT(gt);
        if (!t_enu_map_init) {
            T_enu_map = T_enu_sensor;
            // T_enu_map.block(0, 3, 3, 1) = T_enu_sensor.block(0, 3, 3, 1);
            save_transform(root + "map/T_enu_map.txt", T_enu_map);
            t_enu_map_init = true;
        }
        Eigen::Matrix4d T_map_sensor = get_inverse_tf(T_enu_map) * T_enu_sensor;

        removeMotionDistortion(pc, times, T_enu_sensor, gt);

        // todo: colorize points

        Eigen::Matrix4d prior = T_map_sensor;
        DP newCloud = DP(pc, labels, intensities, desclabels);
        newCloud = removeScanner->filter(newCloud);
        std::string fname;
        get_name_from_file(lidar_files[i], fname);

        if (!map_init) {
            map = newCloud;
            save_transform(root + "map/frame_poses/" + fname + ".txt", T_map_sensor);
            map_init = true;
            std::vector<float> loc = {0.0, 0.0};
            frame_loc.push_back(loc);
            continue;
        }

        // Use GPS as an initial guess (prior) for ICP
        std::cout << "* Starting ICP" << std::endl;

        std::vector<float> loc = {float(T_map_sensor(0, 3)), float(T_map_sensor(1, 3))};
        // Get K closest frames to build submap
        std::vector<int> closestK;
        getClosestKFrames(loc, frame_loc, retrieveK, closestK);
        print_vec(closestK);
        DP submap;
        getSubMap(root, closestK, submap);
        // Downsample to speed up ICP
        submap = randSubsample->filter(submap);

        T = icp(randSubsample->filter(newCloud), submap, prior);

        DP transformed = rigidTrans->compute(newCloud, T);
        transformed.save(root + "map/frames/" + fname + ".ply");
        map.concatenate(transformed);
        save_transform(root + "map/frame_poses/" + fname + ".txt", T);
        std::cout << "* Finished ICP" << std::endl;

        // Downsample map
        // map = densityFilter->filter(map);
        // map = uniformSubsample->filter(map);
        if (i % 20 == 0) {
            std::cout << "* Downsampling map" << std::endl;
            map = ocTreeSubsample->filter(map);
            std::cout << "* Saving map" << std::endl;
            map.save(root + "map/map.ply");
        }
        prev_x = gt[1];
        prev_y = gt[2];
        frame_loc.push_back(loc);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> e = t2 - t1;
        std::cout << "* Frame time: " << e.count() << " seconds" << std::endl;
    }
    map.save(root + "map/map.ply");
}
