#pragma once
#include <Eigen/Dense>
#include <vector>

// Returns in the inverse of a 4x4 homogeneous transform
Eigen::Matrix4d get_inverse_tf(Eigen::Matrix4d T);

/*!
   \brief Enforce orthogonality conditions on the given rotation matrix such that det(R) == 1 and R.tranpose() * R = I
   \param R The input rotation matrix either 2x2 or 3x3, will be overwritten with a slightly modified matrix to
   satisfy orthogonality conditions.
*/
void enforce_orthogonality(Eigen::MatrixXd &R);

/*!
   \brief Returns the output of the cross operator.
        For 3 x 1 input, cross(x) * y is equivalent to cross_product(x, y)
        For 6 x 1 input, x = [rho, phi]^T. out = [cross(phi), rho; 0 0 0 1]
   \param x Input vector which can be 3 x 1 or 6 x 1.
   \return If the input if 3 x 1, the output is 3 x 3, if the input is 6 x 1, the output is 4 x 4.
*/
Eigen::MatrixXd cross(Eigen::VectorXd x);

/*!
   \brief This function converts from a lie vector to a 4 x 4 SE(3) transform.
        Lie Vector xi = [rho, phi]^T (6 x 1) --> SE(3) T = [C, R; 0 0 0 1] (4 x 4)
   \param x Input vector is 6 x 1
   \return Output is 4 x SE(3) transform
*/
Eigen::Matrix4d se3ToSE3(Eigen::MatrixXd xi);

/*!
   \brief Removes motion distortion from a pointcloud.
        Given a pointcloud, a timestamp for each point, a transform from the sensor to the origin,
        and a ground truth data vector, this function adjusts the position of each point so as to remove
        the motion distortion.
   \param pc input/output pointcloud, the position of points will be modified to remove distortion.
   \param times vector of timestamps for each point
   \param T_enu_sensor Global transform from the sensor frame to ENU at timestamp times[0].
   \param gt Vector of ground truth associated with this pointcloud:
        GPSTime,x,y,z,vel_x,vel_y,vel_z,roll,pitch,heading,ang_vel_z
*/
void removeMotionDistortion(Eigen::MatrixXd &pc, std::vector<float> &times, Eigen::Matrix4d T_enu_sensor,
    std::vector<double> gt);

/*!
   \brief Get the K closest keyframes to the specified location.
   \param loc [x, y] location of the query frame.
   \param frame_locs Vector specifying the location of the candidate keyframes.
   \param K The number of keyframes to retrieve. If frame_locs.size() < K, then less will be returned.
   \param closestK output vector of indices specifying the closest K keyframes, not in any particular order.
*/
void getClosestKFrames(std::vector<float> loc, std::vector<std::vector<float>> &frame_locs, uint K,
    std::vector<int> & closestK);

/*!
   \brief Calculates the translation and rotation difference between two 4x4 homogeneous transformations.
        The two transformations should be doing the same thing, ex: T_map_sensor.
*/
void poseError(Eigen::Matrix4d T1, Eigen::Matrix4d T2, double &trans_error, double &rot_error);

/*!
   \brief Given a 3x3 rotation matrix, this function extracts 3-2-1 yaw-pitch-roll angles.
*/
void rotToYawPitchRoll(Eigen::Matrix3d C, double &yaw, double &pitch, double &roll);
