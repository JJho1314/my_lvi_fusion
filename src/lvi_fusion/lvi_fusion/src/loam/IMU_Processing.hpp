#pragma once
#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
// #include <fast_lio/States.h>
#include <geometry_msgs/Vector3.h>
#include "utility.h"
// #include "factor/integration_base.h"
/// *************Preconfiguration
#define MAX_INI_COUNT (20)



extern Eigen::Vector3d vG;


const inline bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);}; // 曲率中存储的是点云的时间戳
bool check_state(StatesGroup &state_inout);
void check_in_out_state(const StatesGroup &state_in, StatesGroup &state_inout);

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
     * 预积分残差
     * 用预积分起止时刻对应的视觉里程计位姿（还包括速度、偏置）变换，与预积分量相减构建残差
    */
    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj) const
    {
        Eigen::Matrix<double, 15, 1> residuals = Eigen::Matrix<double, 15, 1>::Zero();;
        // 预积分结束时刻误差相对于预积分起始时刻误差的微分
        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG); // p对bg的雅克比

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        // 预积分起始时刻，ba的变化量
        Eigen::Vector3d dba = Bai - linearized_ba;
        // 预积分起始时刻，bg的变化量
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        // 预积分起止时间段内的旋转量Q，带噪声修正
        Eigen::Quaterniond corrected_delta_q = dq * Utility::deltaQ(dq_dbg * dbg);
        // 预积分起止时间段内的速度差量V，带噪声修正
        Eigen::Vector3d corrected_delta_v = dv + dv_dba * dba + dv_dbg * dbg;
        // 预积分起止时间段内的平移量P，带噪声修正
        Eigen::Vector3d corrected_delta_p = dp + dp_dba * dba + dp_dbg * dbg;

        // 用预积分起止时刻对应的视觉里程计位姿变换，与预积分量相减构建残差
        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * vG * dtime * dtime + Pj - Pi - Vi * dtime) - corrected_delta_p;
        // 旋转这里只存了四元数的虚部
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (vG * dtime + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }


  ImuProcess();
  ~ImuProcess();

  void Process(const MeasureGroup &meas, StatesGroup &state, PointCloudXYZINormal::Ptr pcl_un_);
  void Reset();
  void IMU_Initial(const MeasureGroup &meas, StatesGroup &state, int &N);

  // Eigen::Matrix3d Exp(const Eigen::Vector3d &ang_vel, const double &dt);

  void IntegrateGyr(const std::vector<sensor_msgs::Imu::ConstPtr> &v_imu);

  void UndistortPcl(const MeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZINormal &pcl_in_out);
  void lic_state_propagate(const MeasureGroup &meas, StatesGroup &state_inout);
  void lic_point_cloud_undistort(const MeasureGroup &meas,  const StatesGroup &state_inout, PointCloudXYZINormal &pcl_out);
  StatesGroup imu_preintegration(const StatesGroup & state_inout, std::deque<sensor_msgs::Imu::ConstPtr> & v_imu,  double end_pose_dt = 0);
  ros::NodeHandle nh;

  void Integrate(const sensor_msgs::ImuConstPtr &imu);
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void processIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity,const Eigen::Vector3d &acc_0, const Eigen::Vector3d &gyr_0);

  Eigen::Vector3d angvel_last;   // 减去bias后的，上一帧用于积分的平均角速度
  Eigen::Vector3d acc_s_last;  // 减去bias和重力后的，上一帧用于积分的平均acc

  Eigen::Matrix<double,DIM_OF_PROC_N,1> cov_proc_noise;

  Eigen::Vector3d cov_acc;
  Eigen::Vector3d cov_gyr;

  // std::ofstream fout;

 public:
  /*** Whether is the first frame, init for first frame ***/
  bool b_first_frame_ = true;
  bool imu_need_init_ = true;

  int init_iter_num = 1;    // 初始化使用的IMU帧数
  Eigen::Vector3d mean_acc;
  Eigen::Vector3d mean_gyr;

  /*** Undistorted pointcloud ***/
  PointCloudXYZINormal::Ptr cur_pcl_un_;

  // For timestamp usage
  sensor_msgs::ImuConstPtr last_imu_;  // 上一个scan处理后的最后一个IMU数据

  /*** For gyroscope integration ***/
  double start_timestamp_;
  /// Making sure the equal size: v_imu_ and v_rot_
  std::deque<sensor_msgs::ImuConstPtr> v_imu_;
  std::vector<Eigen::Matrix3d> v_rot_pcl_;
  std::vector<Pose6D> IMU_pose; // 一个lidar scan中imu积分得到的各个帧时刻的位姿

  // 用于存储滑窗内的数据交换
  // vector<double> dt_buf[(WINDOW_SIZE + 1)];
  // vector<Eigen::Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
  // vector<Eigen::Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

  // // IMU系坐标，限于滑窗内的数据 Twi
  // Eigen::Vector3d        Ps[(WINDOW_SIZE + 1)];
  // Eigen::Vector3d        Vs[(WINDOW_SIZE + 1)];
  // Eigen::Matrix3d        Rs[(WINDOW_SIZE + 1)];
  // Eigen::Vector3d        Bas[(WINDOW_SIZE + 1)];
  // Eigen::Vector3d        Bgs[(WINDOW_SIZE + 1)];

  // IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)]; // 滑动窗口中的预积分器
  // IntegrationBase *tmp_pre_integration;

  std::deque<sensor_msgs::Imu::ConstPtr> QimuMsg; 
  Eigen::Quaterniond dq; // 上一帧到这一帧的激光数据的预积分deltaR
  Eigen::Vector3d dp; // 没有去掉重力
  Eigen::Vector3d dv;
  Eigen::Vector3d linearized_bg; // 上一刻的bg， 认为上一刻到这一刻的bg相同
  Eigen::Vector3d linearized_ba;
  Eigen::Matrix<double, 15, 15> covariance;
  Eigen::Matrix<double, 15, 15> jacobian;
  Eigen::Matrix<double, 18, 18> noise;
  double dtime; // 积分时间
};
