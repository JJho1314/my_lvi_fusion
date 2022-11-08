/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <cmath>
#include <cassert>
#include <cstring>
#include <eigen3/Eigen/Dense>

// 临时添加
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <thread>
#include <mutex>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include "ceres/ceres.h"

#define Lidar_WINDOW_SIZE 2
#define Camera_WINDOW_SIZE 7
#define WINDOW_SIZE 9 // WINDOW_SIZE = Lidar_WINDOW_SIZE + Camera_WINDOW_SIZE
#define COV_OMEGA_NOISE_DIAG 1e-1
#define COV_ACC_NOISE_DIAG 0.4
#define COV_GYRO_NOISE_DIAG 0.2

#define COV_BIAS_ACC_NOISE_DIAG 0.05
#define COV_BIAS_GYRO_NOISE_DIAG 0.1

#define COV_START_ACC_DIAG 1e-1
#define COV_START_GYRO_DIAG 1e-1

// TODO 


extern int frame_count;
enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};

using namespace std;

// int 帧id； int 特征点ID；int 相机id 左目0 右目1；Matrix<< x, y, z, p_u, p_v, velocity_x, velocity_y; 
// extern map<int, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>> lidar_depth_map;
// typedef pcl::PointXYZI PointType;




// extern std::mutex m_odom;
// extern std::deque<nav_msgs::Odometry> odomQueue;//LIO-SAM的预积分里程计
// extern ros::Publisher pub_depth_cloud;

// extern pcl::PointCloud<PointType>::Ptr depthCloud;
// extern std::mutex mtx_lidar;
// extern sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame);
// extern float pointDistance(PointType p);
class Utility
{
  public:
    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
    {
        typedef typename Derived::Scalar Scalar_t;

        Eigen::Quaternion<Scalar_t> dq;
        Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
        half_theta /= static_cast<Scalar_t>(2.0);
        dq.w() = static_cast<Scalar_t>(1.0);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }
    
    // hat()操作
    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
    {
        Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
        ans << typename Derived::Scalar(0), -q(2), q(1),
            q(2), typename Derived::Scalar(0), -q(0),
            -q(1), q(0), typename Derived::Scalar(0);
        return ans;
    }

    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> &q)
    {
        //printf("a: %f %f %f %f", q.w(), q.x(), q.y(), q.z());
        //Eigen::Quaternion<typename Derived::Scalar> p(-q.w(), -q.x(), -q.y(), -q.z());
        //printf("b: %f %f %f %f", p.w(), p.x(), p.y(), p.z());
        //return q.template w() >= (typename Derived::Scalar)(0.0) ? q : Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
        return q;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
    {
        Eigen::Quaternion<typename Derived::Scalar> qq = positify(q);
        Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
        ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
        ans.template block<3, 1>(1, 0) = qq.vec(), ans.template block<3, 3>(1, 1) = qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + skewSymmetric(qq.vec());
        return ans;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p)
    {
        Eigen::Quaternion<typename Derived::Scalar> pp = positify(p);
        Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
        ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
        ans.template block<3, 1>(1, 0) = pp.vec(), ans.template block<3, 3>(1, 1) = pp.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - skewSymmetric(pp.vec());
        return ans;
    }

    /**
     * 旋转矩阵计算Yaw、Pitch、Roll姿态角
    */
    static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
    {
        Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        ypr(0) = y;
        ypr(1) = p;
        ypr(2) = r;

        return ypr / M_PI * 180.0;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr)
    {
        typedef typename Derived::Scalar Scalar_t;

        Scalar_t y = ypr(0) / 180.0 * M_PI;
        Scalar_t p = ypr(1) / 180.0 * M_PI;
        Scalar_t r = ypr(2) / 180.0 * M_PI;

        Eigen::Matrix<Scalar_t, 3, 3> Rz;
        Rz << cos(y), -sin(y), 0,
            sin(y), cos(y), 0,
            0, 0, 1;

        Eigen::Matrix<Scalar_t, 3, 3> Ry;
        Ry << cos(p), 0., sin(p),
            0., 1., 0.,
            -sin(p), 0., cos(p);

        Eigen::Matrix<Scalar_t, 3, 3> Rx;
        Rx << 1., 0., 0.,
            0., cos(r), -sin(r),
            0., sin(r), cos(r);

        return Rz * Ry * Rx;
    }

    static Eigen::Matrix3d g2R(const Eigen::Vector3d &g);

    template <size_t N>
    struct uint_
    {
    };

    template <size_t N, typename Lambda, typename IterT>
    void unroller(const Lambda &f, const IterT &iter, uint_<N>)
    {
        unroller(f, iter, uint_<N - 1>());
        f(iter + N);
    }

    template <typename Lambda, typename IterT>
    void unroller(const Lambda &f, const IterT &iter, uint_<0>)
    {
        f(iter);
    }

    template <typename T>
    static T normalizeAngle(const T& angle_degrees) {
      T two_pi(2.0 * 180);
      if (angle_degrees > 0)
      return angle_degrees -
          two_pi * std::floor((angle_degrees + T(180)) / two_pi);
      else
        return angle_degrees +
            two_pi * std::floor((-angle_degrees + T(180)) / two_pi);
    };
};


class ParamServer
{
public:

    ros::NodeHandle m_ros_node_handle;

    float poseCovThreshold;
    float gpsCovThreshold;
    std::string gpsTopic;
    std::string gpsOdomTopic;
    float mappingProcessInterval;
    float surroundingkeyframeAddingDistThreshold;
    float surroundingkeyframeAddingAngleThreshold;
    std::string intialMethod;
    std::vector<double> initial_Pose;
    bool optimization_with_GPS;
    bool useGpsElevation;
    float gpsAddDis;
    float mappingSurfLeafSize;
    int numberOfCores;
    std::string evalFormat;
    float gpsPoseProportion;
    float globalMapVisualizationSearchRadius;
    bool loopClosureEnableFlag;
    float loopClosureFrequency;
    int historyKeyframeSearchNum;
    float historyKeyframeSearchRadius;
    float historyKeyframeFitnessScore;
    float historyKeyframeSearchTimeDiff;
    int surroundingKeyframeSize;
    float keyframeMeterGap,keyframeDegGap;



    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> m_Rli; //  外参Rli
    Eigen::Matrix<double, 3, 1> m_tli;
    Eigen::Matrix<double, 3, 3> m_Ril; //  外参Rli
    Eigen::Matrix<double, 3, 1> m_til;


    //ROS的参数服务
    ParamServer()
    {
 
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_mapping/keyframeDegGap", keyframeDegGap,10.f );
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_mapping/keyframeMeterGap", keyframeMeterGap,0.5f );

        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/gpsTopic", gpsTopic, std::string("/gps/fix"));
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/gpsOdomTopic", gpsOdomTopic, std::string("/gps/correct_odom"));
        
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/mappingProcessInterval", mappingProcessInterval, 0.1f);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0f);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.1f);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/intialMethod", intialMethod,  std::string("human"));
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/optimization_with_GPS", optimization_with_GPS,  false);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/poseCovThreshold", poseCovThreshold,  0.1f);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/gpsCovThreshold", gpsCovThreshold,  1.0f);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/gpsAddDis", gpsAddDis,  1.0f);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/useGpsElevation", useGpsElevation,  false);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/mappingSurfLeafSize", mappingSurfLeafSize,  0.4f);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/numberOfCores", numberOfCores,  4);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/evalFormat", evalFormat,  std::string("kitti"));
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/gpsPoseProportion", gpsPoseProportion,  1.f);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius,  1000.0f);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/loopClosureEnableFlag", loopClosureEnableFlag,  true);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/loopClosureFrequency", loopClosureFrequency,  1.0f);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/historyKeyframeSearchNum", historyKeyframeSearchNum,  25);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/historyKeyframeSearchRadius", historyKeyframeSearchRadius,  15.0f);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/historyKeyframeFitnessScore", historyKeyframeFitnessScore,  0.5f);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff,  30.0f);
        get_ros_parameter( m_ros_node_handle, "lvi_fusion_common/surroundingKeyframeSize", surroundingKeyframeSize, 50);
        

        m_ros_node_handle.param<vector<double>>( "lvi_fusion_common/initialPose", initial_Pose,  std::vector<double>());

        // lvi_fusion_lio
        std::vector< double > Rli,tli;
        m_ros_node_handle.param<vector<double>>("lvi_fusion_lio/Rli", Rli,  vector<double>());
        m_ros_node_handle.param<vector<double>>("lvi_fusion_lio/tli", tli,  vector<double>());
        m_Rli = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( Rli.data() );
        m_tli = Eigen::Map< Eigen::Matrix< double, 3, 1 > >( tli.data() );

        m_Ril = m_Rli.transpose();
        m_til = -m_Ril*m_tli;

        usleep(100);
    }
public:
    // 读取ros参数
    template < typename T >
    inline T get_ros_parameter( ros::NodeHandle &nh, const std::string parameter_name, T &parameter, T default_val )
    {
        nh.param< T >( parameter_name.c_str(), parameter, default_val );
        // ENABLE_SCREEN_PRINTF;
        cout << "[Ros_parameter]: " << parameter_name << " ==> " << parameter << std::endl;
        return parameter;
    }

    // 当前点到原点的距离
    template < typename PointType >
    inline float pointDistance(PointType p)
    {
        return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
    }

    // 两点间的距离
    template < typename PointType >
    inline float pointDistance(PointType p1, PointType p2)
    {
        return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
    }

    /**
 * @details 发布点云
 * @param thisPub 发布者
 * @param thisCloud 点云指针
 * @param thisStamp 时间戳
 * @param thisFrame 坐标系
 * @return thisCloud对应的ros点云
 */
template < typename PointType >
void publishCloud(ros::Publisher *thisPub, typename pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    
}


};
