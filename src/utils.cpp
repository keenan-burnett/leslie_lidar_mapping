#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include "utils.hpp"

static inline bool exists(const std::string& name) {
    struct stat buffer;
    return !(stat (name.c_str(), &buffer) == 0);
}

struct less_than_img {
    inline bool operator() (const std::string& img1, const std::string& img2) {
        std::vector<std::string> parts;
        boost::split(parts, img1, boost::is_any_of("."));
        int64_t i1 = std::stoll(parts[0]);
        boost::split(parts, img2, boost::is_any_of("."));
        int64_t i2 = std::stoll(parts[0]);
        return i1 < i2;
    }
};

void get_file_names(std::string datadir, std::vector<std::string> &files, std::string extension) {
    DIR *dirp = opendir(datadir.c_str());
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {
        if (exists(dp->d_name)) {
            if (!extension.empty()) {
                std::vector<std::string> parts;
                boost::split(parts, dp->d_name, boost::is_any_of("."));
                if (parts[parts.size() - 1].compare(extension) != 0)
                    continue;
            }
            files.push_back(dp->d_name);
        }
    }
    // Sort files in ascending order of time stamp
    std::sort(files.begin(), files.end(), less_than_img());
}

static float getFloatFromByteArray(char *byteArray, uint index) {
    return *( (float *)(byteArray + index));
}

// Input is a .bin binary file.
void load_velodyne(std::string path, Eigen::MatrixXd &pc, Eigen::MatrixXd & intensities,
    std::vector<float> &times) {
    std::ifstream ifs(path, std::ios::binary);
    std::vector<char> buffer(std::istreambuf_iterator<char>(ifs), {});
    int float_offset = 4;
    int fields = 6;  // x, y, z, i, r, t
    int N = buffer.size() / (float_offset * fields);
    int point_step = float_offset * fields;
    pc = Eigen::MatrixXd::Ones(4, N);
    intensities = Eigen::MatrixXd::Zero(1, N);
    times = std::vector<float>(N);
    int j = 0;
#pragma omp parallel
    for (uint i = 0; i < buffer.size(); i += point_step) {
        pc(0, j) = getFloatFromByteArray(buffer.data(), i);
        pc(1, j) = getFloatFromByteArray(buffer.data(), i + float_offset);
        pc(2, j) = getFloatFromByteArray(buffer.data(), i + float_offset * 2);
        intensities(0, j) = getFloatFromByteArray(buffer.data(), i + float_offset * 3);
        times[j] = getFloatFromByteArray(buffer.data(), i + float_offset * 5);
        j++;
    }
}

void load_image(std::string path, cv::Mat &img) {
    img = cv::imread(path, cv::IMREAD_COLOR);
}

void get_name_from_file(std::string file, std::string &name) {
    std::vector<std::string> parts;
    boost::split(parts, file, boost::is_any_of("."));
    name = parts[0];
}

bool get_groundtruth_data(std::string gtfile, std::string sensor_file, std::vector<double> &gt) {
    std::vector<std::string> farts;
    boost::split(farts, sensor_file, boost::is_any_of("."));
    std::string ftime = farts[0];   // Unique timestamp identifier for the sensor_file to search for
    std::ifstream ifs(gtfile);
    std::string line;
    gt.clear();
    bool gtfound = false;
    std::getline(ifs, line);  // clear out the csv file header before searching
    while (std::getline(ifs, line)) {
        std::vector<std::string> parts;
        boost::split(parts, line, boost::is_any_of(","));
        if (parts[0] == ftime) {
            for (uint i = 1; i < parts.size(); ++i) {
                gt.push_back(std::stod(parts[i]));
            }
            gtfound = true;
            break;
        }
    }
    return gtfound;
}

static Eigen::Matrix3d roll(double r) {
    Eigen::Matrix3d C1 = Eigen::Matrix3d::Identity();
    C1 << 1, 0, 0, 0, cos(r), sin(r), 0, -sin(r), cos(r);
    return C1;
}

static Eigen::Matrix3d pitch(double p) {
    Eigen::Matrix3d C2 = Eigen::Matrix3d::Identity();
    C2 << cos(p), 0, -sin(p), 0, 1, 0, sin(p), 0, cos(p);
    return C2;
}

static Eigen::Matrix3d yaw(double y) {
    Eigen::Matrix3d C3 = Eigen::Matrix3d::Identity();
    C3 << cos(y), sin(y), 0, -sin(y), cos(y), 0, 0, 0, 1;
    return C3;
}

Eigen::Matrix4d getTransformFromGT(std::vector<double> gt) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 3) = gt[1];
    T(1, 3) = gt[2];
    T(2, 3) = gt[3];
    Eigen::Matrix3d C = roll(gt[7]) * pitch(gt[8]) * yaw(gt[9]);
    T.block(0, 0, 3, 3) = C;
    return T;
}

void save_transform(std::string path, Eigen::Matrix4d T) {
    std::ofstream ofs;
    ofs.open(path, std::ios::out);
    ofs << std::setprecision(18);
    ofs << T(0, 0) << " " << T(0, 1) << " " << T(0, 2) << " " << T(0, 3) << "\n";
    ofs << T(1, 0) << " " << T(1, 1) << " " << T(1, 2) << " " << T(1, 3) << "\n";
    ofs << T(2, 0) << " " << T(2, 1) << " " << T(2, 2) << " " << T(2, 3) << "\n";
    ofs << T(3, 0) << " " << T(3, 1) << " " << T(3, 2) << " " << T(3, 3) << "\n";
}

void load_transform(std::string path, Eigen::Matrix4d &T) {
    T = Eigen::Matrix4d::Zero();
    std::ifstream ifs(path);
    std::string line;
    uint i = 0;
    while (std::getline(ifs, line)) {
        if (i >= 4)
            break;
        std::vector<std::string> parts;
        boost::split(parts, line, boost::is_any_of(" "));
        for (uint j = 0; j < 4; ++j) {
            T(i, j) = std::stod(parts[j]);
        }
        i++;
    }
}

bool filter_on_time(std::string fname, std::vector<std::vector<int>> valid_times) {
    std::vector<std::string> parts;
    boost::split(parts, fname, boost::is_any_of("."));
    int time = std::stoll(parts[0]) / 1e9;
    bool keep = false;
    for (auto times : valid_times) {
        if (times[0] <= time && time <= times[1]) {
            keep = true;
            break;
        }
    }
    return keep;
}

static void bilinear_interp(cv::Mat &img, Eigen::Vector4d ubar, Eigen::Vector3d &bgr) {
    bgr = Eigen::Vector3d::Zero();
    double u = ubar(0);
    double v = ubar(1);
    int u1 = floor(u);
    int u2 = ceil(u);
    int v1 = floor(v);
    int v2 = ceil(v);
    for (uint i = 0; i < 3; ++i) {
        double q11 = img.at<cv::Vec3b>(u1, v1)[i];
        double q12 = img.at<cv::Vec3b>(u1, v2)[i];
        double q21 = img.at<cv::Vec3b>(u2, v1)[i];
        double q22 = img.at<cv::Vec3b>(u2, v2)[i];
        double f_y1 = ((u2 - u) / (u2 - u1)) * q11 +  ((u - u1) / (u2 - u1)) * q21;
        double f_y2 = ((u2 - u) / (u2 - u1)) * q12 +  ((u - u1) / (u2 - u1)) * q22;
        double f =    ((v2 - v) / (v2 - v1)) * f_y1 + ((v - v1) / (v2 - v1)) * f_y2;
        bgr(i) = f;
    }
}

void get_point_colors(DP pc, cv::Mat img, Eigen::Matrix4d P, Eigen::MatrixXd T_cam_lidar,
    Eigen::MatrixXd &point_colors) {
    assert(img.channels() == 3);
    uint N = pc.getNbPoints();
    uint H = img.size().height;
    uint W = img.size().width;
    point_colors = Eigen::MatrixXd::Zero(3, N);
#pragma omp parallel
    for (uint i = 0; i < N; ++i) {
        Eigen::Vector4d xbar = pc.features.block(0, i, 4, 1);  // [x, y, z, 1]^T
        xbar = T_cam_lidar * xbar;
        Eigen::Vector4d ubar = xbar / xbar(2);
        ubar = P * ubar;  // [u, v, 1, d]^T
        if (0 <= ubar(0) && ubar(0) <= W - 1 && 0 <= ubar(1) && ubar(1) <= H - 1) {
            Eigen::Vector3d bgr = Eigen::Vector3d::Zero();
            bilinear_interp(img, ubar, bgr);
            point_colors.block(0, i, 3, 1) = bgr;
        }
    }
}

bool get_closest_frame(std::string src_file, std::vector<std::string> &tgt_files, std::string &closest_frame) {
    double min_delta = 1.0;
    bool found = false;
    std::vector<std::string> parts;
    boost::split(parts, src_file, boost::is_any_of("."));
    int64_t src_time = std::stoll(parts[0]);
    for (uint i = 0; i < tgt_files.size(); ++i) {
        boost::split(parts, tgt_files[i], boost::is_any_of("."));
        int64_t t2 = std::stoll(parts[0]);
        double delta = fabs((src_time - t2) / 1.0e9);
        if (delta < min_delta) {
            found = true;
            min_delta = delta;
            closest_frame = tgt_files[i];
        }
    }
    return found;
}

static Eigen::Matrix4d get_inverse_tf(Eigen::Matrix4d T) {
    Eigen::Matrix3d R = T.block(0, 0, 3, 3);
    Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
    T2.block(0, 0, 3, 3) = R.transpose();
    T2.block(0, 3, 3, 1) = -1 * R.transpose() * T.block(0, 3, 3, 1);
    return T2;
}


void colorize_cloud(DP &cloud, Eigen::Matrix4d T_enu_lidar, Eigen::Matrix4d P_cam, std::string lidar_file,
    std::vector<std::string> &cam_files, std::string cam_pose_file, std::string root) {
    // Allocate descriptors for the colors
    cloud.allocateDescriptor("blue", 1);
    cloud.allocateDescriptor("green", 1);
    cloud.allocateDescriptor("red", 1);
    uint start_row = cloud.getDescriptorStartingRow("blue");
    std::string closest_cam;
    // Get the name of the camera file that is closest in time to the lidar_file
    assert(get_closest_frame(lidar_file, cam_files, closest_cam));
    // Retrieve ground truth pose information for the camera file
    std::vector<double> gt;
    assert(get_groundtruth_data(cam_pose_file, closest_cam, gt));
    // Extract the 4x4 homogeneous transform from the ground truth vector
    Eigen::Matrix4d T_enu_cam = getTransformFromGT(gt);
    Eigen::Matrix4d T_cam_lidar = get_inverse_tf(T_enu_cam) * T_enu_lidar;
    // Load the image
    cv::Mat img;
    load_image(root + "camera/" + closest_cam, img);
    // Extract the colors for each point where possible, zero otherwise.
    Eigen::MatrixXd point_colors;
    get_point_colors(cloud, img, P_cam, T_cam_lidar, point_colors);
    // Copy the point_colors into the output cloud object
    assert(cloud.descriptors.rows() >= point_colors.rows() + start_row &&
        cloud.descriptors.cols() == point_colors.cols());
    cloud.descriptors.block(start_row, 0, point_colors.rows(), point_colors.cols()) = point_colors;
}
