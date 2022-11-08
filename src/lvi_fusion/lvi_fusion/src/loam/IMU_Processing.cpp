#include "IMU_Processing.hpp"

// #define COV_NOISE_EXT_I2C_R (0.0 * 1e-3)
// #define COV_NOISE_EXT_I2C_T (0.0 * 1e-3)
// #define COV_NOISE_EXT_I2C_Td (0.0 * 1e-3)

Eigen::Vector3d vG{0.0, 0.0, 9.8};  //g在世界坐标系下的值

double g_lidar_star_tim = 0;
ImuProcess::ImuProcess() : b_first_frame_( true ), imu_need_init_( true ), last_imu_( nullptr ), start_timestamp_( -1 )
{
    Eigen::Quaterniond q( 0, 1, 0, 0 ); // unuse 
    Eigen::Vector3d    t( 0, 0, 0 );

    init_iter_num = 1;
    cov_acc = Eigen::Vector3d( COV_START_ACC_DIAG, COV_START_ACC_DIAG, COV_START_ACC_DIAG );
    cov_gyr = Eigen::Vector3d( COV_START_GYRO_DIAG, COV_START_GYRO_DIAG, COV_START_GYRO_DIAG );
    mean_acc = Eigen::Vector3d( 0, 0, -9.805 );
    mean_gyr = Eigen::Vector3d( 0, 0, 0 );
    angvel_last = Zero3d;
    cov_proc_noise = Eigen::Matrix< double, DIM_OF_PROC_N, 1 >::Zero();

    dq.setIdentity();
    dp.setZero();
    dv.setZero();
    dtime = 0;
    covariance.setZero();
    jacobian.setIdentity();
    linearized_bg.setZero();
    linearized_ba.setZero();
    noise.setZero();
    // TODO 确认一下这儿的noise初始化
    noise.block<3, 3>(0, 0) =  (COV_ACC_NOISE_DIAG * COV_ACC_NOISE_DIAG) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(3, 3) =  (COV_GYRO_NOISE_DIAG * COV_GYRO_NOISE_DIAG) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(6, 6) =  (COV_ACC_NOISE_DIAG * COV_ACC_NOISE_DIAG) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(9, 9) =  (COV_GYRO_NOISE_DIAG * COV_GYRO_NOISE_DIAG) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(12, 12) =  (COV_BIAS_ACC_NOISE_DIAG * COV_BIAS_ACC_NOISE_DIAG) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(15, 15) =  (COV_BIAS_GYRO_NOISE_DIAG * COV_BIAS_GYRO_NOISE_DIAG) * Eigen::Matrix3d::Identity();
    // Lidar_offset_to_IMU = Eigen::Vector3d(0.0, 0.0, -0.0);
    // fout.open(DEBUG_FILE_DIR("imu.txt"),std::ios::out);

    
}

ImuProcess::~ImuProcess()
{ /**fout.close();*/
}

// 重置IMU的参数
void ImuProcess::Reset()
{
    ROS_WARN( "Reset ImuProcess" );
    angvel_last = Zero3d;
    cov_proc_noise = Eigen::Matrix< double, DIM_OF_PROC_N, 1 >::Zero();

    cov_acc = Eigen::Vector3d( COV_START_ACC_DIAG, COV_START_ACC_DIAG, COV_START_ACC_DIAG );
    cov_gyr = Eigen::Vector3d( COV_START_GYRO_DIAG, COV_START_GYRO_DIAG, COV_START_GYRO_DIAG );
    mean_acc = Eigen::Vector3d( 0, 0, -9.805 );
    mean_gyr = Eigen::Vector3d( 0, 0, 0 );

    imu_need_init_ = true;
    b_first_frame_ = true;
    init_iter_num = 1;

    last_imu_ = nullptr;

    // gyr_int_.Reset(-1, nullptr);
    start_timestamp_ = -1;
    v_imu_.clear();
    IMU_pose.clear();

    cur_pcl_un_.reset( new PointCloudXYZINormal() );

    dq.setIdentity();
    dp.setZero();
    dv.setZero();
    dtime = 0;
    covariance.setZero();
    jacobian.setIdentity();
    linearized_bg.setZero();
    linearized_ba.setZero();
}


/**
 * @brief           IMU静态初始化
 * 
 * @param meas        一个scan的测量数据
 * @param state_inout   当前状态state
 * @param N             N初始化使用的IMU帧数
 */
// TODO 这里可以改成我的
void ImuProcess::IMU_Initial( const MeasureGroup &meas, StatesGroup &state_inout, int &N )
{
    /** 1. initializing the gravity, gyro bias, acc and gyro covariance
     ** 2. normalize the acceleration measurenments to unit gravity **/
    ROS_INFO( "IMU Initializing: %.1f %%", double( N ) / MAX_INI_COUNT * 100 );
    Eigen::Vector3d cur_acc, cur_gyr;

    if ( b_first_frame_ )
    {
        Reset();
        N = 1;
        b_first_frame_ = false;
    }

    for ( const auto &imu : meas.imu )
    {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

        mean_acc += ( cur_acc - mean_acc ) / N;
        mean_gyr += ( cur_gyr - mean_gyr ) / N;

        cov_acc = cov_acc * ( N - 1.0 ) / N + ( cur_acc - mean_acc ).cwiseProduct( cur_acc - mean_acc ) * ( N - 1.0 ) / ( N * N );
        cov_gyr = cov_gyr * ( N - 1.0 ) / N + ( cur_gyr - mean_gyr ).cwiseProduct( cur_gyr - mean_gyr ) * ( N - 1.0 ) / ( N * N );
        // cov_acc = Eigen::Vector3d(0.1, 0.1, 0.1);
        // cov_gyr = Eigen::Vector3d(0.01, 0.01, 0.01);
        N++;
    }

    // TODO: fix the cov
    cov_acc = Eigen::Vector3d( COV_START_ACC_DIAG, COV_START_ACC_DIAG, COV_START_ACC_DIAG );
    cov_gyr = Eigen::Vector3d( COV_START_GYRO_DIAG, COV_START_GYRO_DIAG, COV_START_GYRO_DIAG );
    state_inout.gravity = Eigen::Vector3d( 0, 0, 9.805 );
    state_inout.rot_end = Eye3d;
    state_inout.bias_g = mean_gyr;
}

//imu预积分更新状态state_inout
void ImuProcess::lic_state_propagate( const MeasureGroup &meas, StatesGroup &state_inout )
{
    /*** add the imu of the last frame-tail to the of current frame-head ***/
    auto v_imu = meas.imu;
    v_imu.push_front( last_imu_ ); // 好像扔了两次last_IMU_,其实没有，因为v_imu是临时变量
    // const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
    const double &imu_end_time = v_imu.back()->header.stamp.toSec();
    const double &pcl_beg_time = meas.lidar_beg_time;

    /*** sort point clouds by offset time ***/
    // TODO 去畸变的时候已经排过一次了
    PointCloudXYZINormal pcl_out = *( meas.lidar );
    std::sort( pcl_out.points.begin(), pcl_out.points.end(), time_list );
    const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double( 1000 );
    double        end_pose_dt = pcl_end_time - imu_end_time;

    state_inout = imu_preintegration( state_inout, v_imu, end_pose_dt );
    last_imu_ = meas.imu.back();
}

// Avoid abnormal state input
// 判断任意方向的速度是否大于30，如果大于10置0并return false
bool check_state( StatesGroup &state_inout )
{
    bool is_fail = false;
    for ( int idx = 0; idx < 3; idx++ )
    {
        if ( fabs( state_inout.vel_end( idx ) ) > 30 )  //TODO xyz任意方向速度大于10
        {
            is_fail = true;
            scope_color( ANSI_COLOR_RED_BG );
            for ( int i = 0; i < 10; i++ )
            {
                cout << __FILE__ << ", " << __LINE__ << ", check_state fail !!!! " << state_inout.vel_end.transpose() << endl;
            }
            state_inout.vel_end( idx ) = 0.0;
        }
    }
    return is_fail;
}

// Avoid abnormal state input
void check_in_out_state( const StatesGroup &state_in, StatesGroup &state_inout )
{
    if ( ( state_in.pos_end - state_inout.pos_end ).norm() > 1.0 )
    {
        scope_color( ANSI_COLOR_RED_BG );
        for ( int i = 0; i < 10; i++ )
        {
            cout << __FILE__ << ", " << __LINE__ << ", check_in_out_state fail !!!! " << state_in.pos_end.transpose() << " | "
                 << state_inout.pos_end.transpose() << endl;
        }
        state_inout.pos_end = state_in.pos_end;
    }
}

std::mutex g_imu_premutex;

/**
 * @brief IMU预积分得到更新state_in，得到初始位姿估计
 * 
 * @param state_in      积分前的状态，以及积分后的状态
 * @param v_imu         一个scan的imu数据
 * @param end_pose_dt   点云end点和最后一个iMU数据的时间间隔（最后一小段时间单独处理）
 * @return StatesGroup 
 */
StatesGroup ImuProcess::imu_preintegration( const StatesGroup &state_in, std::deque< sensor_msgs::Imu::ConstPtr > &v_imu, double end_pose_dt )
{
    std::unique_lock< std::mutex > lock( g_imu_premutex );
    StatesGroup                    state_inout = state_in;
    if ( check_state( state_inout ) ) // 判断任意方向的速度是否大于10，如果大于10置0并return false
    {
        state_inout.display( state_inout, "state_inout" );
        state_in.display( state_in, "state_in" );
    }
    Eigen::Vector3d acc_imu( 0, 0, 0 ), angvel_avr( 0, 0, 0 ), acc_avr( 0, 0, 0 ), vel_imu( 0, 0, 0 ), pos_imu( 0, 0, 0 );
    vel_imu = state_inout.vel_end;
    pos_imu = state_inout.pos_end;
    Eigen::Matrix3d R_imu( state_inout.rot_end );
    Eigen::MatrixXd F_x( Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES >::Identity() );// process matrix = A
    Eigen::MatrixXd cov_w( Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES >::Zero() ); // noise matrix = B
    double          dt = 0;
    int             if_first_imu = 1;
    // printf("IMU start_time = %.5f, end_time = %.5f, state_update_time = %.5f, start_delta = %.5f\r\n", v_imu.front()->header.stamp.toSec() -
    // g_lidar_star_tim,
    //        v_imu.back()->header.stamp.toSec() - g_lidar_star_tim,
    //        state_in.last_update_time - g_lidar_star_tim,
    //        state_in.last_update_time - v_imu.front()->header.stamp.toSec());

    // 遍历所有IMU数据，中值积分
    for ( std::deque< sensor_msgs::Imu::ConstPtr >::iterator it_imu = v_imu.begin(); it_imu != ( v_imu.end() - 1 ); it_imu++ )
    {
        // if(g_lidar_star_tim == 0 || state_inout.last_update_time == 0)
        // {
        //   return state_inout;
        // }
        sensor_msgs::Imu::ConstPtr head = *( it_imu );
        sensor_msgs::Imu::ConstPtr tail = *( it_imu + 1 );

        Eigen::Vector3d acc_0(head->linear_acceleration.x,head->linear_acceleration.y,head->linear_acceleration.z);
        Eigen::Vector3d acc_1(tail->linear_acceleration.x,tail->linear_acceleration.y,tail->linear_acceleration.z);

        angvel_avr << 0.5 * ( head->angular_velocity.x + tail->angular_velocity.x ), 0.5 * ( head->angular_velocity.y + tail->angular_velocity.y ),
            0.5 * ( head->angular_velocity.z + tail->angular_velocity.z );
        acc_avr << 0.5 * ( head->linear_acceleration.x + tail->linear_acceleration.x ),
            0.5 * ( head->linear_acceleration.y + tail->linear_acceleration.y ), 0.5 * ( head->linear_acceleration.z + tail->linear_acceleration.z );

        angvel_avr -= state_inout.bias_g;

        Eigen::Vector3d  un_acc_0 = dq * (acc_0 -  state_inout.bias_a);

        acc_avr = acc_avr - state_inout.bias_a;

        if ( tail->header.stamp.toSec() < state_inout.last_update_time )
        {
            continue;
        }

        if ( if_first_imu )
        {
            if_first_imu = 0;
            dt = tail->header.stamp.toSec() - state_inout.last_update_time;
        }
        else
        {
            dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        }
        if ( dt > 0.05 ) // TODO 这里应该是异常
        {
            dt = 0.05;
        }
        double dt2 = dt*dt;
        /* covariance propagation */
        Eigen::Matrix3d acc_avr_skew;
        Eigen::Matrix3d Exp_f = Exp( angvel_avr, dt );  //  一帧IMU数据的deltaR   
        Eigen::Matrix3d m3dR = dq.matrix()*Exp_f; // 更新后的值
        Eigen::Vector3d un_acc_1 = m3dR* (acc_1 - state_inout.bias_a);
        auto un_acc_avr = 0.5 * (un_acc_0 + un_acc_1);
        dp += dv*dt + 0.5*un_acc_avr*dt2;
        dv += un_acc_avr*dt;

        acc_avr_skew << SKEW_SYM_MATRIX( acc_avr );

    
        // Eigen::Matrix3d Jr_omega_dt = right_jacobian_of_rotion_matrix<double>(angvel_avr*dt);
        Eigen::Matrix3d Jr_omega_dt = Eigen::Matrix3d::Identity();
        F_x.block< 3, 3 >( 0, 0 ) = Exp_f.transpose();
        // F_x.block<3, 3>(0, 9) = -Eye3d * dt;
        F_x.block< 3, 3 >( 0, 9 ) = -Jr_omega_dt * dt;
        // F_x.block<3,3>(3,0)  = -R_imu * off_vel_skew * dt;
        F_x.block< 3, 3 >( 3, 3 ) = Eye3d; // Already the identity.
        F_x.block< 3, 3 >( 3, 6 ) = Eye3d * dt;
        F_x.block< 3, 3 >( 6, 0 ) = -R_imu * acc_avr_skew * dt;
        F_x.block< 3, 3 >( 6, 12 ) = -R_imu * dt;
        F_x.block< 3, 3 >( 6, 15 ) = Eye3d * dt;

        Eigen::Matrix3d cov_acc_diag, cov_gyr_diag, cov_omega_diag;
        cov_omega_diag = Eigen::Vector3d( COV_OMEGA_NOISE_DIAG, COV_OMEGA_NOISE_DIAG, COV_OMEGA_NOISE_DIAG ).asDiagonal();
        cov_acc_diag = Eigen::Vector3d( COV_ACC_NOISE_DIAG, COV_ACC_NOISE_DIAG, COV_ACC_NOISE_DIAG ).asDiagonal();
        cov_gyr_diag = Eigen::Vector3d( COV_GYRO_NOISE_DIAG, COV_GYRO_NOISE_DIAG, COV_GYRO_NOISE_DIAG ).asDiagonal();
        // cov_w.block<3, 3>(0, 0) = cov_omega_diag * dt * dt;
        cov_w.block< 3, 3 >( 0, 0 ) = Jr_omega_dt * cov_omega_diag * Jr_omega_dt * dt * dt;
        cov_w.block< 3, 3 >( 3, 3 ) = R_imu * cov_gyr_diag * R_imu.transpose() * dt * dt;
        cov_w.block< 3, 3 >( 6, 6 ) = cov_acc_diag * dt * dt;
        cov_w.block< 3, 3 >( 9, 9 ).diagonal() =
            Eigen::Vector3d( COV_BIAS_GYRO_NOISE_DIAG, COV_BIAS_GYRO_NOISE_DIAG, COV_BIAS_GYRO_NOISE_DIAG ) * dt * dt; // bias gyro covariance
        cov_w.block< 3, 3 >( 12, 12 ).diagonal() =
            Eigen::Vector3d( COV_BIAS_ACC_NOISE_DIAG, COV_BIAS_ACC_NOISE_DIAG, COV_BIAS_ACC_NOISE_DIAG ) * dt * dt; // bias acc covariance

        // cov_w.block<3, 3>(18, 18).diagonal() = Eigen::Vector3d(COV_NOISE_EXT_I2C_R, COV_NOISE_EXT_I2C_R, COV_NOISE_EXT_I2C_R) * dt * dt; // bias
        // gyro covariance cov_w.block<3, 3>(21, 21).diagonal() = Eigen::Vector3d(COV_NOISE_EXT_I2C_T, COV_NOISE_EXT_I2C_T, COV_NOISE_EXT_I2C_T) * dt
        // * dt;  // bias acc covariance cov_w(24, 24) = COV_NOISE_EXT_I2C_Td * dt * dt;

        Eigen::Vector3d a_0_x = acc_0 - state_inout.bias_a;
        Eigen::Vector3d a_1_x = acc_1 - state_inout.bias_a;
        Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;
        R_w_x<< SKEW_SYM_MATRIX(angvel_avr);
        R_a_0_x<< SKEW_SYM_MATRIX(a_0_x);
        R_a_1_x<< SKEW_SYM_MATRIX(a_1_x);
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(15, 15);
        A.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        A.block<3, 3>(0, 3) = -0.25 * dq.toRotationMatrix() * R_a_0_x * dt2 + 
                            -0.25 * m3dR * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * dt) * dt2;
        A.block<3, 3>(0, 6) = Eigen::MatrixXd::Identity(3,3) * dt;
        A.block<3, 3>(0, 9) = -0.25 * (dq.toRotationMatrix() + m3dR) * dt2;
        A.block<3, 3>(0, 12) = -0.25 * m3dR * R_a_1_x * dt2 * -dt;
        A.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_w_x * dt;
        A.block<3, 3>(3, 12) = -1.0 * Eigen::MatrixXd::Identity(3,3) * dt;
        A.block<3, 3>(6, 3) = -0.5 * dq.toRotationMatrix() * R_a_0_x * dt + 
                            -0.5 * m3dR * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * dt) * dt;
        A.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
        A.block<3, 3>(6, 9) = -0.5 * (dq.toRotationMatrix() + m3dR) * dt;
        A.block<3, 3>(6, 12) = -0.5 * m3dR * R_a_1_x * dt * -dt;
        A.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
        A.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();

        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(15,18);
        B.block<3, 3>(0, 0) =  0.25 * dq.toRotationMatrix() * dt2;
        B.block<3, 3>(0, 3) =  0.25 * -m3dR * R_a_1_x  * dt2* 0.5 * dt;
        B.block<3, 3>(0, 6) =  0.25 * m3dR * dt2;
        B.block<3, 3>(0, 9) =  B.block<3, 3>(0, 3);
        B.block<3, 3>(3, 3) =  0.5 * Eigen::MatrixXd::Identity(3,3) * dt;
        B.block<3, 3>(3, 9) =  0.5 * Eigen::MatrixXd::Identity(3,3) * dt;
        B.block<3, 3>(6, 0) =  0.5 * dq.toRotationMatrix() * dt;
        B.block<3, 3>(6, 3) =  0.5 * -m3dR * R_a_1_x  * dt * 0.5 * dt;
        B.block<3, 3>(6, 6) =  0.5 * m3dR* dt;
        B.block<3, 3>(6, 9) =  B.block<3, 3>(6, 3);
        B.block<3, 3>(9, 12) = Eigen::MatrixXd::Identity(3,3) * dt;
        B.block<3, 3>(12, 15) = Eigen::MatrixXd::Identity(3,3) * dt;
        jacobian = A * jacobian;
        covariance = A * covariance * A.transpose() + B * noise * B.transpose();



        state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

        R_imu = R_imu * Exp_f;
        acc_imu = R_imu * acc_avr - state_inout.gravity;
        pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
        vel_imu = vel_imu + acc_imu * dt;
        angvel_last = angvel_avr;
        acc_s_last = acc_imu;

        Eigen::Quaterniond qtmp(m3dR);
        if (qtmp.w()<0)
            qtmp.coeffs() *= -1;
        dq = qtmp.normalized();
        dtime += dt;

        // cout <<  std::setprecision(3) << " dt = " << dt << ", acc: " << acc_avr.transpose()
        //      << " acc_imu: " << acc_imu.transpose()
        //      << " vel_imu: " << vel_imu.transpose()
        //      << " omega: " << angvel_avr.transpose()
        //      << " pos_imu: " << pos_imu.transpose()
        //       << endl;
        // cout << "Acc_avr: " << acc_avr.transpose() << endl;
    }

    // cout <<__FILE__ << ", " << __LINE__ <<" ,diagnose lio_state = " << std::setprecision(2) <<(state_inout - StatesGroup()).transpose() << endl;
    /*** calculated the pos and attitude prediction at the frame-end 最后一帧特殊处理***/
    dt = end_pose_dt;

    state_inout.last_update_time = v_imu.back()->header.stamp.toSec() + dt;
    // cout << "Last update time = " <<  state_inout.last_update_time - g_lidar_star_tim << endl;
    if ( dt > 0.1 )
    {
        scope_color( ANSI_COLOR_RED_BOLD );
        for ( int i = 0; i < 1; i++ )
        {
            cout << __FILE__ << ", " << __LINE__ << "dt = " << dt << endl;
        }
        dt = 0.1;
    }
    state_inout.vel_end = vel_imu + acc_imu * dt;
    state_inout.rot_end = R_imu * Exp( angvel_avr, dt );
    state_inout.pos_end = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

    // cout <<__FILE__ << ", " << __LINE__ <<" ,diagnose lio_state = " << std::setprecision(2) <<(state_inout - StatesGroup()).transpose() << endl;

    // cout << "Preintegration State diff = " << std::setprecision(2) << (state_inout - state_in).head<15>().transpose()
    //      <<  endl;
    // std::cout << __FILE__ << " " << __LINE__ << std::endl;
    // check_state(state_inout);
    if ( 0 )
    {
        if ( check_state( state_inout ) )
        {
            // printf_line;
            std::cout << __FILE__ << " " << __LINE__ << std::endl;
            state_inout.display( state_inout, "state_inout" );
            state_in.display( state_in, "state_in" );
        }
        check_in_out_state( state_in, state_inout );
    }
    // cout << (state_inout - state_in).transpose() << endl;
    return state_inout;
}

/**
 * @brief 点云去运动畸变，全部投影到最后一个点的lidar坐标下。
 *        TODO只补偿了旋转，没有补偿平移
 * 
 * @param meas         一个scan的测量数据
 * @param _state_inout 当前状态
 * @param pcl_out       去畸变后的点云
 */
void ImuProcess::lic_point_cloud_undistort( const MeasureGroup &meas, const StatesGroup &_state_inout, PointCloudXYZINormal &pcl_out )
{
    StatesGroup state_inout = _state_inout;
    auto        v_imu = meas.imu; // 当前scan时间中的IMU数据
    v_imu.push_front( last_imu_ );
    const double &imu_end_time = v_imu.back()->header.stamp.toSec();
    const double &pcl_beg_time = meas.lidar_beg_time;
    /*** sort point clouds by offset time 把点云按时间戳从小到大排列***/
    pcl_out = *( meas.lidar );
    std::sort( pcl_out.points.begin(), pcl_out.points.end(), time_list );
    const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double( 1000 );
    /*std::cout << "[ IMU Process ]: Process lidar from " << pcl_beg_time - g_lidar_star_tim << " to " << pcl_end_time- g_lidar_star_tim << ", "
              << meas.imu.size() << " imu msgs from " << imu_beg_time- g_lidar_star_tim << " to " << imu_end_time- g_lidar_star_tim
              << ", last tim: " << state_inout.last_update_time- g_lidar_star_tim << std::endl;
    */
    /*** Initialize IMU pose ***/
    IMU_pose.clear();
    // IMUpose.push_back(set_pose6d(0.0, Zero3d, Zero3d, state.vel_end, state.pos_end, state.rot_end));
    // 把上一lidar结束时的pose也保存进来
    IMU_pose.push_back( set_pose6d( 0.0, acc_s_last, angvel_last, state_inout.vel_end, state_inout.pos_end, state_inout.rot_end ) );

    /*** forward propagation at each imu point ***/
    Eigen::Vector3d acc_imu, 
                    angvel_avr, // 平均角速度
                    acc_avr,  // 平均加速度
                    vel_imu( state_inout.vel_end ),    // 初始速度，随IMU积分更新
                    pos_imu( state_inout.pos_end );  // 初始位置

    Eigen::Matrix3d R_imu( state_inout.rot_end ); // 初始姿态
    Eigen::MatrixXd F_x( Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES >::Identity() ); // F_x，cov_w这两个参数这里没有用到
    Eigen::MatrixXd cov_w( Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES >::Zero() );
    double          dt = 0; // 两帧间隔时间
    // 遍历
    for ( auto it_imu = v_imu.begin(); it_imu != ( v_imu.end() - 1 ); it_imu++ )
    {
        auto &&head = *( it_imu );
        auto &&tail = *( it_imu + 1 );

        // 中值        
        angvel_avr << 0.5 * ( head->angular_velocity.x + tail->angular_velocity.x ), 0.5 * ( head->angular_velocity.y + tail->angular_velocity.y ),
            0.5 * ( head->angular_velocity.z + tail->angular_velocity.z );
        acc_avr << 0.5 * ( head->linear_acceleration.x + tail->linear_acceleration.x ),
            0.5 * ( head->linear_acceleration.y + tail->linear_acceleration.y ), 0.5 * ( head->linear_acceleration.z + tail->linear_acceleration.z );



        angvel_avr -= state_inout.bias_g;
        acc_avr = acc_avr - state_inout.bias_a;

#ifdef DEBUG_PRINT
// fout<<head->header.stamp.toSec()<<" "<<angvel_avr.transpose()<<" "<<acc_avr.transpose()<<std::endl;
#endif
        dt = tail->header.stamp.toSec() - head->header.stamp.toSec();

        // TODO new add
        Eigen::Vector3d acc_0,acc_1, gry_0,gry_1;
        acc_0<< head->linear_acceleration.x,head->linear_acceleration.y,head->linear_acceleration.z;
        acc_1<< tail->linear_acceleration.x,tail->linear_acceleration.y,tail->linear_acceleration.z;
        gry_0<< head->angular_velocity.x,head->angular_velocity.y,head->angular_velocity.z;
        gry_1<< tail->angular_velocity.x,tail->angular_velocity.y,tail->angular_velocity.z;
        // processIMU(dt,acc_1,gry_1,acc_0,gry_0);
        /* covariance propagation */

        Eigen::Matrix3d acc_avr_skew;
        Eigen::Matrix3d Exp_f = Exp( angvel_avr, dt );
        acc_avr_skew << SKEW_SYM_MATRIX( acc_avr );
#ifdef DEBUG_PRINT
// fout<<head->header.stamp.toSec()<<" "<<angvel_avr.transpose()<<" "<<acc_avr.transpose()<<std::endl;
#endif
        // 积分IMU数据，更新状态
        /* propagation of IMU attitude */
        R_imu = R_imu * Exp_f;

        /* Specific acceleration (global frame) of IMU */
        acc_imu = R_imu * acc_avr - state_inout.gravity;

        /* propagation of IMU */
        pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

        /* velocity of IMU */
        vel_imu = vel_imu + acc_imu * dt;

        /* save the poses at each IMU measurements */
        angvel_last = angvel_avr;
        acc_s_last = acc_imu;
        double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
        // std::cout<<"acc "<<acc_imu.transpose()<<"vel "<<acc_imu.transpose()<<"vel "<<pos_imu.transpose()<<std::endl;
        // 保存积分得到的位姿
        IMU_pose.push_back( set_pose6d( offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu ) );
    }

    /*** calculated the pos and attitude prediction at the frame-end 这里假设最后这段时间加速度和角速度不变***/
    dt = pcl_end_time - imu_end_time;
    // TODO new add 最后一帧单独处理一下
    Eigen::Vector3d acc_0,acc_1, gry_0,gry_1;
    auto && tmp = *(v_imu.end() );
    acc_1<< tmp->linear_acceleration.x,tmp->linear_acceleration.y,tmp->linear_acceleration.z;  
    gry_1<< tmp->angular_velocity.x,tmp->angular_velocity.y,tmp->angular_velocity.z;
    // processIMU(dt,acc_1,gry_1,acc_0,gry_0);

    state_inout.vel_end = vel_imu + acc_imu * dt;
    state_inout.rot_end = R_imu * Exp( angvel_avr, dt );
    state_inout.pos_end = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
    // IMU_frame -> lidar_frame 最后一个点在世界坐标系下的位置
    // Twl = Twi * Til
    Eigen::Vector3d pos_liD_e = state_inout.pos_end + state_inout.rot_end * Lidar_offset_to_IMU;
    // auto R_liD_e   = state_inout.rot_end * Lidar_R_to_IMU;

#ifdef DEBUG_PRINT
    std::cout << "[ IMU Process ]: vel " << state_inout.vel_end.transpose() << " pos " << state_inout.pos_end.transpose() << " ba"
              << state_inout.bias_a.transpose() << " bg " << state_inout.bias_g.transpose() << std::endl;
    std::cout << "propagated cov: " << state_inout.cov.diagonal().transpose() << std::endl;
#endif

    /*** undistort each lidar point (backward propagation) ***/
    auto it_pcl = pcl_out.points.end() - 1;
    for ( auto it_kp = IMU_pose.end() - 1; it_kp != IMU_pose.begin(); it_kp-- )
    {
        auto head = it_kp - 1;
        R_imu << MAT_FROM_ARRAY( head->rot );
        acc_imu << VEC_FROM_ARRAY( head->acc );
        // std::cout<<"head imu acc: "<<acc_imu.transpose()<<std::endl;
        vel_imu << VEC_FROM_ARRAY( head->vel );
        pos_imu << VEC_FROM_ARRAY( head->pos );
        angvel_avr << VEC_FROM_ARRAY( head->gyr );

        for ( ; it_pcl->curvature / double( 1000 ) > head->offset_time; it_pcl-- )
        {
            dt = it_pcl->curvature / double( 1000 ) - head->offset_time;

            //! TODO 注意lvi_fusion认为lidar和IMU旋转外参为E
            // e -> 表示最后一个点, i表示第i个点
            /* Transform to the 'end' frame, using only the rotation
             * Note: Compensation direction is INVERSE of Frame's moving direction
             * So if we want to compensate a point at timestamp-i to the frame-e
             * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame
             * R
             *  */
            Eigen::Matrix3d R_i( R_imu * Exp( angvel_avr, dt ) ); // Rwl_i ： i时刻
            // pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt + R_i * Lidar_offset_to_IMU = 点i在世界坐标系下的位置 
            // - pos_liD_e = 和最后一个点的位置差 
            Eigen::Vector3d T_ei( pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt + R_i * Lidar_offset_to_IMU - pos_liD_e );

            Eigen::Vector3d P_i( it_pcl->x, it_pcl->y, it_pcl->z );
           // 补偿到最后一个点时刻的lidar做标系下
            Eigen::Vector3d P_compensate = state_inout.rot_end.transpose() * ( R_i * P_i + T_ei );

            /// save Undistorted points and their rotation
            it_pcl->x = P_compensate( 0 );
            it_pcl->y = P_compensate( 1 );
            it_pcl->z = P_compensate( 2 );

            if ( it_pcl == pcl_out.points.begin() )
                break;
        }
    }
}

/**
 * @brief   0、如果是第一帧IMU，初始化IMU
 *          1、点云去畸变
 *          2、IMU预积分更新状态
 * @param meas         一个scan的lidar 和 IMU数据
 * @param stat          当前状态
 * @param cur_pcl_un_  去畸变后的点云
 */
void ImuProcess::Process( const MeasureGroup &meas, StatesGroup &stat, PointCloudXYZINormal::Ptr cur_pcl_un_ )
{
    // double t1, t2, t3;
    // t1 = omp_get_wtime();

    if ( meas.imu.empty() )
    {
        // std::cout << "no imu data" << std::endl;
        return;
    };
    ROS_ASSERT( meas.lidar != nullptr );

    // IMU没有初始化就先初始化
    if ( imu_need_init_ )
    {
        /// The very first lidar frame
        IMU_Initial( meas, stat, init_iter_num );

        imu_need_init_ = true;

        last_imu_ = meas.imu.back();

        if ( init_iter_num > MAX_INI_COUNT )
        {
            imu_need_init_ = false;
            // std::cout<<"mean acc: "<<mean_acc<<" acc measures in word frame:"<<state.rot_end.transpose()*mean_acc<<std::endl;
            ROS_INFO(
                "IMU Initials: Gravity: %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",
                stat.gravity[ 0 ], stat.gravity[ 1 ], stat.gravity[ 2 ], stat.bias_g[ 0 ], stat.bias_g[ 1 ], stat.bias_g[ 2 ], cov_acc[ 0 ],
                cov_acc[ 1 ], cov_acc[ 2 ], cov_gyr[ 0 ], cov_gyr[ 1 ], cov_gyr[ 2 ] );
        }

        return;
    }

    /// Undistort points： the first point is assummed as the base frame
    /// Compensate lidar points with IMU rotation (with only rotation now)
    // if ( 0 || (stat.last_update_time < 0.1))

    if ( 0 )
    {
        // UndistortPcl(meas, stat, *cur_pcl_un_);
    }
    else
    {
        if ( 1 )
        {
            lic_point_cloud_undistort( meas, stat, *cur_pcl_un_ );
        }
        else // 不去畸变
        {
            *cur_pcl_un_ = *meas.lidar;
        }
        lic_state_propagate( meas, stat );
    }
    // t2 = omp_get_wtime();

    last_imu_ = meas.imu.back();

    // t3 = omp_get_wtime();

    // std::cout<<"[ IMU Process ]: Time: "<<t3 - t1<<std::endl;
}


/**
 * 处理一帧IMU，积分
 * 用前一图像帧位姿，前一图像帧与当前图像帧之间的IMU数据，积分计算得到当前图像帧位姿
 * Rs，Ps，Vs
 * @param t                     当前时刻
 * @param dt                    与前一帧时间间隔
 * @param linear_acceleration   当前时刻加速度
 * @param angular_velocity      当前时刻角速度
 * @param acc_0                 上一帧的
 * @param gyr_0    
 * @param ba
 * @param bg
*/
// void ImuProcess::processIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity,const Eigen::Vector3d &acc_0, const Eigen::Vector3d &gyr_0)
// {
// #if 1


//     // 滑窗中保留11帧，frame_count表示现在处理第几帧，一般处理到第11帧时就保持不变了
//     // 如果滑动窗口中的预积分器对象不存在就创建
//     if (!pre_integrations[frame_count])
//     {

//         pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

//     }
//      // 在初始化时，第一帧图像特征点数据没有对应的预积分，因此要从第二帧开始
//     if (frame_count != 0)
//     {
//         // 当前帧预积分器,添加前一图像帧与当前图像帧之间的IMU数据
//         pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
//         //if(solver_flag != NON_LINEAR)
//             // 和上面一样的操作，这个量用来做初始化用的
//             tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

//         // 缓存IMU数据 （用于滑窗交换数据）
//         dt_buf[frame_count].push_back(dt);
//         linear_acceleration_buf[frame_count].push_back(linear_acceleration);
//         angular_velocity_buf[frame_count].push_back(angular_velocity);

//         // 又是一个中值积分，更新滑窗中状态量，本质是给非线性优化提供可信的初始值
//         // 用IMU数据进行积分，当积完一个measurement中所有IMU数据后，就得到了对应图像帧在世界坐标系中的Ps、Vs、Rs
//         // 下面这一部分的积分，在没有成功完成初始化时似乎是没有意义的，因为在没有成功初始化时，对IMU数据来说是没有世界坐标系的
//         // 当成功完成了初始化后，下面这一部分积分才有用，它可以通过IMU积分得到滑动窗口中最新帧在世界坐标系中的
//         int j = frame_count;  
//         // 前一时刻加速度 
//         Eigen::Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - vG;
//         // 中值积分，用前一时刻与当前时刻角速度平均值，对时间积分
//         Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
//         // 当前时刻姿态Q
//         Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
//         // 当前时刻加速度
//         Eigen::Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - vG;
//         // 中值积分，用前一时刻与当前时刻加速度平均值，对时间积分
//         Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
//         // 当前时刻位置P
//         Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
//         // 当前时刻速度V
//         Vs[j] += dt * un_acc;
//     }

// #endif
// }