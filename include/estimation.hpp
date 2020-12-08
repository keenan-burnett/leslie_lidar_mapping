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
   // Lie Vector xi = [rho, phi]^T (6 x 1) --> SE(3) T = [C, R; 0 0 0 1] (4 x 4)
   \param x Input vector is 6 x 1
   \return Output is 4 x SE(3) transform
*/
Eigen::Matrix4d se3ToSE3(Eigen::MatrixXd xi);

// Given a pointcloud, a timestamp for each point, a transform from the sensor to the origin,
// and a ground truth data vector, this function adjusts the position of each point so as to remove
// the motion distortion.
// Note1: points must be in the 'sensor' frame of T_enu_sensor
// Note2: T_enu_sensor must be defined for point 0 (pc[:, 0])
void removeMotionDistortion(Eigen::MatrixXd &pc, std::vector<float> &times, Eigen::Matrix4d T_enu_sensor,
    std::vector<double> gt);
