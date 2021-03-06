#include <iostream>
#include <iomanip>
#include "estimation.hpp"
#include "nabo/nabo.h"

Eigen::Matrix4d get_inverse_tf(Eigen::Matrix4d T) {
    Eigen::Matrix3d R = T.block(0, 0, 3, 3);
    Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
    T2.block(0, 0, 3, 3) = R.transpose();
    T2.block(0, 3, 3, 1) = -1 * R.transpose() * T.block(0, 3, 3, 1);
    return T2;
}

void enforce_orthogonality(Eigen::MatrixXd &R) {
    if (R.cols() == 3) {
        const Eigen::Vector3d col1 = R.block(0, 1, 3, 1).normalized();
        const Eigen::Vector3d col2 = R.block(0, 2, 3, 1).normalized();
        const Eigen::Vector3d newcol0 = col1.cross(col2);
        const Eigen::Vector3d newcol1 = col2.cross(newcol0);
        R.block(0, 0, 3, 1) = newcol0;
        R.block(0, 1, 3, 1) = newcol1;
        R.block(0, 2, 3, 1) = col2;
    } else if (R.cols() == 2) {
        const double epsilon = 0.001;
        if (fabs(R(0, 0) - R(1, 1)) > epsilon || fabs(R(1, 0) + R(0, 1)) > epsilon) {
            std::cout << "ERROR: this is not a proper rigid transformation!" << std::endl;
        }
        double a = (R(0, 0) + R(1, 1)) / 2;
        double b = (-R(1, 0) + R(0, 1)) / 2;
        double sum = sqrt(pow(a, 2) + pow(b, 2));
        a /= sum;
        b /= sum;
        R(0, 0) = a; R(0, 1) = b;
        R(1, 0) = -b; R(1, 1) = a;
    }
}

Eigen::MatrixXd cross(Eigen::VectorXd x) {
    Eigen::MatrixXd X;
    assert(x.rows() == 3 || x.rows() == 6);
    if (x.rows() == 3) {
        X = Eigen::MatrixXd::Zero(3, 3);
        X << 0, -x(2), x(1),
             x(2), 0, -x(0),
             -x(1), x(0), 0;
    } else {
        X = Eigen::MatrixXd::Zero(4, 4);
        X << 0, -x(5), x(4), x(0),
             x(5), 0, -x(3), x(1),
             -x(4), x(3), 0, x(2),
             0, 0, 0, 1;
    }
    return X;
}

Eigen::Matrix4d se3ToSE3(Eigen::MatrixXd xi) {
    assert(xi.rows() == 6);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::Vector3d rho = xi.block(0, 0, 3, 1);
    Eigen::Vector3d phibar = xi.block(3, 0, 3, 1);
    double phi = phibar.norm();
    Eigen::MatrixXd C = Eigen::MatrixXd::Identity(3, 3);
    if (phi != 0) {
        phibar.normalize();
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        C = cos(phi) * I + (1 - cos(phi)) * phibar * phibar.transpose() + sin(phi) * cross(phibar);
        enforce_orthogonality(C);
        Eigen::Matrix3d J = I * sin(phi) / phi + (1 - sin(phi) / phi) * phibar * phibar.transpose() +
            cross(phibar) * (1 - cos(phi)) / phi;
        rho = J * rho;
    }
    T.block(0, 0, 3, 3) = C;
    T.block(0, 3, 3, 1) = rho;
    return T;
}

static int get_closest(std::vector<float> vec, float value) {
    int closest = 0;
    float mind = 10000;
    for (uint i = 0; i < vec.size(); ++i) {
        float d = fabs(vec[i] - value);
        if (d < mind) {
            mind = d;
            closest = i;
        }
    }
    return closest;
}

void removeMotionDistortion(Eigen::MatrixXd &pc, std::vector<float> &times, Eigen::Matrix4d T_enu_sensor,
    std::vector<double> gt) {
    std::cout << "* Removing motion distortion" << std::endl;
    float t0 = times[0];  // Time at which the transform was obtained
    Eigen::Vector3d vbar_enu = {gt[4], gt[5], gt[6]};
    Eigen::Matrix3d C_sens_enu = get_inverse_tf(T_enu_sensor).block(0, 0, 3, 3);
    Eigen::Vector3d vbar_sens = C_sens_enu * vbar_enu;
    Eigen::MatrixXd varpi = Eigen::MatrixXd::Zero(6, 1);
    varpi.block(0, 0, 3, 1) = vbar_sens;
    varpi(5) = gt[10];

    // Compute T_undistort for several discrete points along the scan
    uint M = 100;
    if (times.size() < M)
        M = times.size();
    std::vector<Eigen::Matrix4d> T_undistort_vec(M);
    std::vector<float> delta_t_vec(M);

    double min_delta_t = 0, max_delta_t = 0;
    for (uint i = 1; i < times.size(); ++i) {
        float delta_t = times[i] - t0;
        if (delta_t < min_delta_t)
            min_delta_t = delta_t;
        if (delta_t > max_delta_t)
            max_delta_t = delta_t;
    }
    for (uint i = 0; i < M; ++i) {
        delta_t_vec[i] = min_delta_t + i * (max_delta_t - min_delta_t) / M;
        T_undistort_vec[i] = se3ToSE3(varpi * delta_t_vec[i]);
    }
    for (uint i = 1; i < times.size(); ++i) {
        float delta_t = times[i] - t0;
        int idx = get_closest(delta_t_vec, delta_t);
        pc.block(0, i, 4, 1) = T_undistort_vec[idx] * pc.block(0, i, 4, 1);
    }
}

void getClosestKFrames(std::vector<float> loc, std::vector<std::vector<float>> &frame_loc, uint K,
    std::vector<int> & closestK) {
    double max_distance = pow(300, 2);
    if (frame_loc.size() <= K) {
        closestK.clear();
        for (uint i = 0; i < frame_loc.size(); ++i) {
            closestK.push_back(i);
        }
        return;
    }
    // copy vector to eigen matrix
    Eigen::MatrixXf M = Eigen::MatrixXf::Zero(3, frame_loc.size());
    const int kk = K;
    for (uint i = 0; i < frame_loc.size(); ++i) {
        M(0, i) = frame_loc[i][0];
        M(1, i) = frame_loc[i][1];
    }
    // Query point: latest frame
    Eigen::VectorXf q = Eigen::VectorXf::Zero(3);
    q(0) = loc[0];
    q(1) = loc[1];
    // Create a KD-Tree for M
    Nabo::NNSearchF* nns = Nabo::NNSearchF::createKDTreeLinearHeap(M);
    // Find K nearest neighbors of the query
    Eigen::VectorXi indices(kk);
    Eigen::VectorXf dists(kk);
    nns->knn(q, indices, dists, kk);
    // Copy results to output vector
    closestK.clear();
    for (int i = 0; i < kk; i++) {
        if (dists(i) < max_distance)
            closestK.push_back(indices(i));
    }
    // cleanup KD-tree
    delete nns;
}

void poseError(Eigen::Matrix4d T1, Eigen::Matrix4d T2, double &trans_error, double &rot_error) {
    Eigen::Matrix4d Terr = get_inverse_tf(T1) * T2;
    trans_error = sqrt(pow(Terr(0, 3), 2) + pow(Terr(1, 3), 2) + pow(Terr(2, 3), 2));
    // std::cout << "x: " << Terr(0, 3) << " y: " << Terr(1, 3) << " z: " << Terr(2, 3) << std::endl;
    double trace = 0;
    for (uint i = 0; i < 3; ++i) {
        trace += Terr(i, i);
    }
    rot_error = acos((trace - 1) / 2);
}

void rotToYawPitchRoll(Eigen::Matrix3d C, double &yaw, double &pitch, double &roll) {
    double eps = 1.0e-15;
    int i = 2, j = 1, k = 0;
    double c_y = sqrt(pow(C(i, i), 2) + pow(C(j, i), 2));
    if (c_y > eps) {
        roll = atan2f(C(j, i), C(i, i));
        pitch = atan2f(-C(k, i), c_y);
        yaw = atan2f(C(k, j), C(k, k));
    } else {
        roll = 0;
        pitch = atan2f(-C(k, i), c_y);
        yaw = atan2f(-C(j, k), C(j, j));
    }
}

void filterPointCloud(Eigen::MatrixXd &pc, double xmin, double xmax, double ymin, double ymax, double zmin,
    double zmax) {
    uint j = 0;
    for (uint i = 0; i < pc.cols(); ++i) {
        if (xmin <= pc(0, i) && pc(0, i) <= xmax &&
            ymin <= pc(1, i) && pc(1, i) <= ymax &&
            zmin <= pc(2, i) && pc(2, i) <= zmax) {
            pc.block(0, j, 4, 1) = pc.block(0, i, 4, 1);
            j++;
        }
    }
    pc.conservativeResize(4, j);
}
