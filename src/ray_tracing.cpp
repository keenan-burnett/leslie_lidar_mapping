// This ray-tracing code is adapted from Hugues Thomas' project.
#include <math.h>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <set>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <boost/algorithm/string.hpp>
#include "ray_tracing.hpp"
#include "estimation.hpp"
#include "utils.hpp"
#include "pointmatcher/PointMatcher.h"

static Eigen::Vector3d max_point(Eigen::MatrixXd & pc) {
    Eigen::Vector3d maxP = Eigen::Vector3d::Zero();
    for (uint i = 0; i < pc.cols(); ++i) {
        if (pc(0, i) > maxP(0))
            maxP(0) = pc(0, i);
        if (pc(1, i) > maxP(1))
            maxP(1) = pc(1, i);
        if (pc(2, i) > maxP(2))
            maxP(2) = pc(2, i);
    }
    return maxP;
}

static Eigen::Vector3d min_point(Eigen::MatrixXd & pc) {
    Eigen::Vector3d minP = Eigen::Vector3d::Zero();
    for (uint i = 0; i < pc.cols(); ++i) {
        if (pc(0, i) < minP(0))
            minP(0) = pc(0, i);
        if (pc(1, i) < minP(1))
            minP(1) = pc(1, i);
        if (pc(2, i) < minP(2))
            minP(2) = pc(2, i);
    }
    return minP;
}

static void cart2pol_(Eigen::MatrixXd &cart_frame, Eigen::MatrixXd &polar_frame) {
    polar_frame = Eigen::MatrixXd::Ones(cart_frame.rows(), cart_frame.cols());
    for (uint i = 0; i < cart_frame.cols(); ++i) {
        Eigen::Vector3d p = cart_frame.block(0, i, 3, 1);
        float rho = p.norm();
        float phi = atan2f(p(1), p(0));
        float theta = atan2f(sqrt(p(0) * p(0) + p(1) * p(1)), p(2));
        polar_frame(0, i) = rho;
        polar_frame(1, i) = theta;
        polar_frame(2, i) = phi + M_PI / 2;
    }
}

static Eigen::Vector3d cart2pol(Eigen::Vector3d &p) {
    float rho = p.norm();
    float phi = atan2f(p(1), p(0));
    float theta = atan2f(sqrt(p(0) * p(0) + p(1) * p(1)), p(2));
    return Eigen::Vector3d(rho, theta, phi + M_PI / 2);
}

static void compare_map_to_frame(Eigen::MatrixXd &aligned_frame, Eigen::MatrixXd &map_points,
    Eigen::MatrixXd &map_normals, Eigen::Matrix4d T_sensor_map, std::unordered_map<VoxKey, size_t> &map_samples,
    float theta_dl, float phi_dl, float map_dl, std::vector<float> &movable_probs, std::vector<int> &movable_counts) {

    float inv_theta_dl = 1.0 / theta_dl;
    float inv_phi_dl = 1.0 / phi_dl;
    float inv_map_dl = 1.0 / map_dl;
    float max_angle = 5 * M_PI / 12;
    float min_vert_cos = cos(M_PI / 3);

    // Mask of the map points not updated yet
    std::vector<bool> not_updated(map_points.cols(), true);

    // Get limits
    Eigen::Vector3d min_P = min_point(aligned_frame) - Eigen::Vector3d(map_dl, map_dl, map_dl);
    Eigen::Vector3d max_P = max_point(aligned_frame) + Eigen::Vector3d(map_dl, map_dl, map_dl);

    // Update full voxels
    // Loop over aligned_frame
    VoxKey k0, k;
    for (uint i = 0; i < aligned_frame.cols(); ++i) {
        // Corresponding key
        k0.x = (int)floor(aligned_frame(0, i) * inv_map_dl);
        k0.y = (int)floor(aligned_frame(1, i) * inv_map_dl);
        k0.z = (int)floor(aligned_frame(2, i) * inv_map_dl);
        // Update the adjacent cells
        for (k.x = k0.x - 1; k.x < k0.x + 2; k.x++) {
            for (k.y = k0.y - 1; k.y < k0.y + 2; k.y++) {
                for (k.z = k0.z - 1; k.z < k0.z + 2; k.z++) {
                    // Update count and movable at this point
                    if (map_samples.count(k) > 0) {
                        // Only update once
                        size_t i0 = map_samples[k];
                        if (not_updated[i0]) {
                            not_updated[i0] = false;
                            movable_counts[i0] += 1;
                        }
                    }
                }
            }
        }
    }

    // Create the free frustum grid

    // get frame in polar coordinates
    Eigen::MatrixXd frame = T_sensor_map * aligned_frame;
    Eigen::MatrixXd polar_frame;
    cart2pol_(frame, polar_frame);

    // Get grid limits
    Eigen::Vector3d minCorner = min_point(polar_frame);
    Eigen::Vector3d maxCorner = max_point(polar_frame);
    Eigen::Vector3d originCorner = minCorner - Eigen::Vector3d(0, 0.5 * theta_dl, 0.5 * phi_dl);

    // Dimensions of the grid
    size_t grid_n_theta = (size_t)floor((maxCorner(1) - originCorner(1)) / theta_dl) + 1;
    size_t grid_n_phi = (size_t)floor((maxCorner(2) - originCorner(2)) / phi_dl) + 1;

    // Initialize variables
    std::vector<float> frustrum_radiuses(grid_n_theta * grid_n_phi, -1.0);

    // Fill the frustrum radiuses
    for (uint i = 0; i < polar_frame.cols(); ++i) {
        // Corresponding key
        Eigen::Vector3d p = polar_frame.block(0, i, 3, 1);
        // Position of point in grid
        size_t i_theta = (size_t)floor((p(1) - originCorner(1)) * inv_theta_dl);
        size_t i_phi = (size_t)floor((p(2) - originCorner(2)) * inv_phi_dl);
        size_t gridIdx = i_theta + grid_n_theta * i_phi;

        // Update the radius in cell
        if (frustrum_radiuses[gridIdx] < 0)
            frustrum_radiuses[gridIdx] = p(0);
        else if (p(0) < frustrum_radiuses[gridIdx])
            frustrum_radiuses[gridIdx] = p(0);
    }

    // Apply margin to free ranges
    float margin = map_dl;
    float frustrum_alpha = theta_dl / 2;
    for (auto &r : frustrum_radiuses) {
        float adapt_margin = r * frustrum_alpha;
        if (margin < adapt_margin)
            r -= adapt_margin;
        else
            r -= margin;
    }

    // Apply frustum casting

    // update free pixels
    float min_r = 2 * map_dl;
    Eigen::Matrix3d C_sensor_map = T_sensor_map.block(0, 0, 3, 3);

    for (uint i = 0; i < map_points.cols(); ++i) {
        // Ignore points updated just now
        if (!not_updated[i])
            continue;
        Eigen::Vector4d p = map_points.block(0, i, 4, 1);
        // Ignore points outside area of the frame
        if (p(0) > max_P(0) || p(1) > max_P(1) || p(2) > max_P(2) ||
            p(0) < min_P(0) || p(1) < min_P(1) || p(2) < min_P(2)) {
            continue;
        }

        // Align point in frame coordinates (and normal)
        p = T_sensor_map * p;
        Eigen::Vector3d xyz = p.block(0, 0, 3, 1);
        Eigen::Vector3d nxyz = C_sensor_map * map_normals.block(0, i, 3, 1);

        // Project in polar coordinates
        Eigen::Vector3d rtp = cart2pol(xyz);

        // Position of point in grid
        size_t i_theta = (size_t)floor((rtp(1) - originCorner(1)) * inv_theta_dl);
        size_t i_phi = (size_t)floor((rtp(2) - originCorner(2)) * inv_phi_dl);
        size_t gridIdx = i_theta + grid_n_theta * i_phi;

        // Update movable prob
        if (rtp(0) > min_r && rtp(0) < frustrum_radiuses[gridIdx]) {
            // Do not update if normal is horizontal and perpendicular to ray (to avoid removing walls)
            if (abs(nxyz(2)) > min_vert_cos) {
                movable_counts[i] += 1;
                movable_probs[i] += 1.0;
            } else {
                float angle = acos(std::min(fabs(xyz.dot(nxyz) / rtp(0)), 1.0));
                if (angle < max_angle) {
                    movable_counts[i] += 1;
                    movable_probs[i] += 1.0;
                }
            }
        }
    }
}

static void getNameFromPath(std::string path, std::string &name) {
    std::vector<std::string> parts;
    boost::split(parts, path, boost::is_any_of("/"));
    std::string endy_bit = parts[parts.size() - 1];
    boost::split(parts, endy_bit, boost::is_any_of("."));
    name = parts[0];
}

int main(int argc, const char *argv[]) {
    std::string root, config;
    std::string new_map_name = "map_no_movable.ply";
    validateArgs(argc, argv, root, config);
    // Params:
    float map_dl = 0.15;
    float theta_dl = 1.29 * M_PI / 180;
    float phi_dl = 0.1 * M_PI / 180;

    // Init point map
    // todo: load points into a vector or modify code to use Eigen matrices directly.
    std::cout << "Loading map..." << std::endl;
    DP map = DP::load(root + "map/map.ply");
    std::cout << "Finished loading map" << std::endl;

    // Create the pointmap voxels
    std::unordered_map<VoxKey, size_t> map_samples;
    uint N = map.features.cols();
    map_samples.reserve(N);
    float inv_map_dl = 1.0 / map_dl;
    VoxKey k0;

    for (uint i = 0; i < N; ++i) {
        k0.x = (int)floor(map.features(0, i) * inv_map_dl);
        k0.y = (int)floor(map.features(1, i) * inv_map_dl);
        k0.z = (int)floor(map.features(2, i) * inv_map_dl);
        if (map_samples.count(k0) < 1) {
            map_samples.emplace(k0, i);
        }
    }

    // Init map movable probabilities and counts
    std::vector<float> movable_probs(N, 0);
    std::vector<int> movable_counts(N, 0);

    // Calculate normal vectors for the map
    std::shared_ptr<PM::DataPointsFilter> normalFilter = PM::get().DataPointsFilterRegistrar.create(
        "SurfaceNormalDataPointsFilter", {{"knn", toParam(10)}, {"epsilon", toParam(5)}, {"keepNormals", toParam(1)},
        {"keepDensities", toParam(0)}});

    std::cout << "Calculating normals for the entire map..." << std::endl;
    map = normalFilter->filter(map);
    std::cout << "Finished calculating normals" << std::endl;
    uint norm_row = map.getDescriptorStartingRow("normals");
    Eigen::MatrixXd map_normals = map.descriptors.block(norm_row, 0, 3, N);

    // Start movable detection

    std::vector<std::string> frame_names;
    getMapFrames(root, frame_names);

    for (uint i = 0; i < frame_names.size(); ++i) {
        std::cout << "Ray tracing frame " << i << " / " << frame_names.size() - 1 << std::endl;
        // Load frame / ply file
        DP frame = DP::load(frame_names[i]);
        // Load pose for this frame
        Eigen::Matrix4d T_map_sensor = Eigen::Matrix4d::Identity();
        std::string name;
        getNameFromPath(frame_names[i], name);
        load_transform(root + "map/frame_poses/" + name + ".txt", T_map_sensor);
        Eigen::Matrix4d T_sensor_map = get_inverse_tf(T_map_sensor);
        // Perform ray-tracing to identity movable points
        compare_map_to_frame(frame.features, map.features, map_normals, T_sensor_map,
            map_samples, theta_dl, phi_dl, map_dl, movable_probs, movable_counts);
    }

    for (uint i = 0; i < movable_probs.size(); ++i) {
        movable_probs[i] = movable_probs[i] / (movable_counts[i] + 1e-6);
        if (movable_counts[i] < 1e-6)
            movable_probs[i] = -1;
    }
    map.allocateDescriptor("movable", 1);
    uint movable_row = map.getDescriptorStartingRow("movable");
    for (uint i = 0; i < movable_probs.size(); ++i) {
	map.descriptors(movable_row, i) = movable_probs[i];
    }
    std::cout << "Adding movable descriptor to the existing map" << std::endl;
    map.save(root + "map/map.ply"); 

    std::cout << "Removing movable points from the map..." << std::endl;
    uint feat_dim = map.features.rows();
    map.removeDescriptor("normals");  // Get rid of normals
    uint desc_dim = map.descriptors.rows();
    uint j = 0;
    for (uint i = 0; i < movable_probs.size(); ++i) {
        if (movable_probs[i] < 0.9) {
            map.features.block(0, j, feat_dim, 1) = map.features.block(0, i, feat_dim, 1);
            map.descriptors.block(0, j, desc_dim, 1) = map.descriptors.block(0, i, desc_dim, 1);
            j++;
        }
    }
    map.conservativeResize(j);

    std::cout << "Saving the filtered map..." << std::endl;
    map.save(root + "map/" + new_map_name);
}
