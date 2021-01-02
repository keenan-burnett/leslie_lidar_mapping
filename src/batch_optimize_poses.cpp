#include <iostream>
#include <fstream>
#include <lgmath.hpp>
#include <steam.hpp>
#include "estimation.hpp"
#include "utils.hpp"

struct RelMeas {
  unsigned int idxA;  // index of pose variable A
  unsigned int idxB;  // index of pose variable B
  lgmath::se3::Transformation meas_T_BA;  // measured transform from A to B
};

int main(int argc, const char *argv[]) {
    std::string root, config;
    if (validateArgs(argc, argv, root, config) != 0) {
        return 1;
    }
    std::vector<std::string> lidar_files;
    get_file_names(root + "lidar/", lidar_files, "bin");
    std::string lidar_pose_file = root + "applanix/lidar_poses.csv";
    std::string lidar_odom_file = root + "applanix/lidar_odom_poses.csv";
    std::string lidar_opti_file = root + "applanix/lidar_optimized_poses.csv";
    std::ofstream ofs;
    ofs.open(lidar_opti_file, std::ios::out);
    ofs << "ROSTime,GPSTime,x,y,z,vel_x,vel_y,vel_z,roll,pitch,heading,ang_vel_z\n";
    ofs.close();

    std::vector<RelMeas> measCollection;

    // Note: pose 0 is the map frame (Identity)

    // Load in the absolute pose measurements from the GPS ground truth
    Eigen::Matrix4d T_enu_map = Eigen::Matrix4d::Identity();
    // uint num_lidar = 9829;
    uint num_lidar = 250;
    uint start = 92;
    for (uint i = start; i < start + num_lidar; ++i) {
        std::cout << "loading abs: " << i << " / " << num_lidar - 1 << std::endl;
        std::vector<double> gt;
        // GPSTime,x,y,z,vel_x,vel_y,vel_z,roll,pitch,heading,ang_vel_z
        assert(get_groundtruth_data(lidar_pose_file, lidar_files[i], gt));
        Eigen::Matrix4d T_enu_sensor = getTransformFromGT(gt);
        if (i == start) {
            T_enu_map.block(0, 3, 3, 1) = T_enu_sensor.block(0, 3, 3, 1);
            save_transform(root + "map/T_enu_map.txt", T_enu_map);
        }
        Eigen::Matrix4d T_map_sensor = get_inverse_tf(T_enu_map) * T_enu_sensor;
        // Create absolute measurement
        RelMeas meas;
        meas.idxA = i + 1 - start;
        meas.idxB = 0;
        meas.meas_T_BA = lgmath::se3::Transformation(T_map_sensor);
        measCollection.push_back(meas);
    }

    // Load in the relative pose measurements from the LIDAR odometry results
    for (uint i = start; i < start + num_lidar - 1; ++i) {
        std::cout << "loading rel: " << i << " / " << num_lidar - 1 << std::endl;
        std::vector<double> gt;
        // TIME1,TIME2,x,y,z,yaw,pitch,roll
        assert(get_odom_data(lidar_odom_file, lidar_files[i], lidar_files[i + 1], gt));
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T(0, 3) = gt[0];
        T(1, 3) = gt[1];
        T(2, 3) = gt[2];
        T.block(0, 0, 3, 3) = roll(gt[5]) * pitch(gt[4]) * yaw(gt[3]);
        // Create relative measurement
        RelMeas meas;
        meas.idxA = i + 1 - start;
        meas.idxB = i + 2 - start;
        meas.meas_T_BA = lgmath::se3::Transformation(get_inverse_tf(T));
        measCollection.push_back(meas);
    }
    uint numPoses = num_lidar + 1;  // One pose for each lidar frame and one for the map frame

    // steam state variables
    std::vector<steam::se3::TransformStateVar::Ptr> poses;

    // Setup state variables - initialized at identity
    for (uint i = 0; i < numPoses; ++i) {
        steam::se3::TransformStateVar::Ptr temp(new steam::se3::TransformStateVar());
        poses.push_back(temp);
    }

    // steam cost terms
    steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());

    // Setup shared noise and loss functions
    float rel_noise_factor = 0.0008;
    float abs_noise_factor = 0.0004;
    steam::BaseNoiseModel<6>::Ptr relNoiseModel(new steam::StaticNoiseModel<6>(rel_noise_factor *
        Eigen::MatrixXd::Identity(6, 6)));
    steam::BaseNoiseModel<6>::Ptr absNoiseModel(new steam::StaticNoiseModel<6>(abs_noise_factor *
        Eigen::MatrixXd::Identity(6, 6)));
    steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

    // Lock first pose (otherwise entire solution is 'floating')
    //  **Note: alternatively we could add a prior (UnaryTransformError) to the first pose.
    poses[0]->setLock(true);

    // Turn measurements into cost terms
    for (uint i = 0; i < measCollection.size(); ++i) {
        // Get first referenced state variable
        steam::se3::TransformStateVar::Ptr& stateVarA = poses[measCollection[i].idxA];

        // Get second referenced state variable
        steam::se3::TransformStateVar::Ptr& stateVarB = poses[measCollection[i].idxB];

        // Get transform measurement
        lgmath::se3::Transformation& meas_T_BA = measCollection[i].meas_T_BA;

        // Construct error function
        steam::TransformErrorEval::Ptr errorfunc(new steam::TransformErrorEval(meas_T_BA, stateVarB, stateVarA));

        // Create cost term and add to problem
        steam::BaseNoiseModel<6>::Ptr noiseModel;
        if (i < num_lidar) {
            noiseModel = absNoiseModel;
        } else {
            noiseModel = relNoiseModel;
        }
        steam::WeightedLeastSqCostTerm<6, 6>::Ptr cost(new steam::WeightedLeastSqCostTerm<6, 6>
            (errorfunc, noiseModel, sharedLossFunc));
        costTerms->add(cost);
    }

    ///
    /// Make Optimization Problem
    ///

    // Initialize problem
    steam::OptimizationProblem problem;

    // Add state variables
    for (uint i = 1; i < poses.size(); ++i) {
        problem.addStateVariable(poses[i]);
    }

    // Add cost terms
    problem.addCostTerm(costTerms);

    ///
    /// Setup Solver and Optimize
    ///
    typedef steam::LevMarqGaussNewtonSolver SolverType;

    // Initialize parameters (enable verbose mode)
    SolverType::Params params;
    params.verbose = true;
    params.maxIterations = 10000;

    // Make solver
    SolverType solver(&problem, params);

    // Optimize
    std::cout << "optimizing..." << std::endl;
    solver.optimize();

    std::cout << "writing output to file" << std::endl;
    ofs.open(lidar_opti_file, std::ios::app);
    for (uint i = start + 1; i < start + numPoses; ++i) {
        std::cout << "writing to file: " << i << " / " << num_lidar - 1 << std::endl;
        std::vector<double> gt;
        // GPSTime,x,y,z,vel_x,vel_y,vel_z,roll,pitch,heading,ang_vel_z
        assert(get_groundtruth_data(lidar_pose_file, lidar_files[i - 1], gt));
        Eigen::Matrix4d T_enu_sensor = getTransformFromGT(gt);
        Eigen::Matrix4d T_map_sensor = get_inverse_tf(T_enu_map) * T_enu_sensor;

        Eigen::Matrix4d T = poses[i - start]->getValue().matrix();
        T = get_inverse_tf(T);

        double trans_error = 0, rot_error = 0;
        poseError(T, T_map_sensor, trans_error, rot_error);
        std::cout << "t_err: " << trans_error << " r_err: " << rot_error << std::endl;

        double yaw = 0, pitch = 0, roll = 0;
        Eigen::Matrix3d C = T.block(0, 0, 3, 3);
        rotToYawPitchRoll(C, yaw, pitch, roll);

        std::string rostime;
        get_name_from_file(lidar_files[i - 1], rostime);

        ofs << rostime << "," << gt[0] << "," << T(0, 3) << "," << T(1, 3) << "," << T(2, 3) << ",";
        ofs << gt[4] << "," << gt[5] << "," << gt[6] << ",";
        ofs << roll << "," << pitch << "," << yaw << "," << gt[10] << "\n";
    }
    ofs.close();

    return 0;
}
