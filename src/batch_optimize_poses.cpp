#include <iostream>
#include <lgmath.hpp>
#include <steam.hpp>

struct RelMeas {
  unsigned int idxA;  // index of pose variable A
  unsigned int idxB;  // index of pose variable B
  lgmath::se3::Transformation meas_T_BA;  // measured transform from A to B
};

int main(int argc, char *argv[]) {
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
    for (uint i = 0; i < lidar_files.size(); ++i) {
        std::vector<double> gt;
        // GPSTime,x,y,z,vel_x,vel_y,vel_z,roll,pitch,heading,ang_vel_z
        assert(get_groundtruth_data(lidar_pose_file, lidar_files[i], gt));
        Eigen::Matrix4d T_enu_sensor = getTransformFromGT(gt);
        removeMotionDistortion(pc, times, T_enu_sensor, gt);
        if (i == 0) {
            T_enu_map.block(0, 3, 3, 1) = T_enu_sensor.block(0, 3, 3, 1);
            save_transform(root + "map/T_enu_map.txt", T_enu_map);
        }
        Eigen::Matrix4d T_map_sensor = get_inverse_tf(T_enu_map) * T_enu_sensor;
        // Create absolute measurement
        RelMeas meas;
        meas.idxA = i + 1;
        meas.idxB = 0;
        meas.meas_T_BA = T_map_sensor;
        measCollection.push_back(meas);
    }

    // Load in the relative pose measurements from the LIDAR odometry results
    for (uint i = 0; i < lidar_files.size() - 1; ++i) {
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
        meas.idxA = i + 1;
        meas.idxB = i + 2;
        meas.meas_T_BA = T;
        measCollection.push_back(meas);
    }

    uint numPoses = lidar_files.size() + 1;  // One pose for each lidar frame and one for the map frame

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
    steam::BaseNoiseModel<6>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<6>(Eigen::MatrixXd::Identity(6, 6)));
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
        steam::WeightedLeastSqCostTerm<6, 6>::Ptr cost(new steam::WeightedLeastSqCostTerm<6, 6>
            (errorfunc, sharedNoiseModel, sharedLossFunc));
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

    // Make solver
    SolverType solver(&problem, params);

    // Optimize
    solver.optimize();

    ofs.open(lidar_opti_file, std::ios::app);
    for (uint i = 1; i < numPoses; ++i) {
        std::vector<double> gt;
        // GPSTime,x,y,z,vel_x,vel_y,vel_z,roll,pitch,heading,ang_vel_z
        assert(get_groundtruth_data(lidar_pose_file, lidar_files[i - 1], gt));

        Eigen::Matrix4d T = poses[i]->getValue();

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
