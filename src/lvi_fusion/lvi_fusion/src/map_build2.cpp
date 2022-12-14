// TODO 本code创新点：用关键帧队列存储点云，避免了在整个地图里搜索点云
#include "utility.h"
#include "Scancontext.h"
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

std::string odometryFrame = "world";
using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
// struct PointXYZIRPYT
// {
//     PCL_ADD_POINT4D
//     PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
//     float roll;
//     float pitch;
//     float yaw;
//     double time;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
// } EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

// POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
//                                    (float, x, x) (float, y, y)
//                                    (float, z, z) (float, intensity, intensity)
//                                    (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
//                                    (double, time, time))

#if 0
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))
#endif
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, timestamp, timestamp))

typedef PointXYZIRPYT  PointTypePose;
typedef pcl::PointXYZI PointType;

class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate; // 初始位姿估计
    Values optimizedEstimate; // 优化后的位姿
    ISAM2 *isam;  // 优化器推理算法
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance; // 优化后的上一关键帧位姿协方差
    Eigen::Quaterniond Qil;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
//     ros::Publisher pubLaserOdometryIncremental;
//     ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
//     ros::Publisher pubRecentKeyFrames;
//     ros::Publisher pubRecentKeyFrame;
//     ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;

    ros::Subscriber subOdom; // 输入当前的里程计
    ros::Subscriber subCloud; // 输入去完畸变后的点云
    ros::Subscriber subGPS;   // 输入gps
//     ros::Subscriber subLoop;  // 输入闭环
    // 改成指针会更高效
    std::deque<nav_msgs::Odometry> gpsQueue; // GPS队列
    std::deque<nav_msgs::Odometry> odomQueue; // odom队列
    std::deque<sensor_msgs::PointCloud2> cloudQueue; // GPS队列  
//     lio_sam::cloud_info cloudInfo;  // 用来存储topic接收的点云

//     vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames; // 当某一帧被选为关键帧之后，他的scan经过降采样作为cornerCloudKeyFrames
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D; //gtsam优化后的地图关键帧位置(x，y，z)
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;//优化后的地图关键帧位置（x，y，z，R, P,Y，time）
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

//     pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // topic接收到的角点点云,当前点云 corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // topic接收到的平面点云surf feature set from odoOptimization
//     pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
//     pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization
//     pcl::PointCloud<PointType>::Ptr laserCloudInRaw; // 当前真在被处理的原始点云

//     pcl::PointCloud<PointType>::Ptr laserCloudOri; // 经过筛选的可以用于匹配的点
//     pcl::PointCloud<PointType>::Ptr coeffSel;   // 优化方向的向量的系数

//     std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
//     std::vector<PointType> coeffSelCornerVec;
//     std::vector<bool> laserCloudOriCornerFlag;
//     std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
//     std::vector<PointType> coeffSelSurfVec;
//     std::vector<bool> laserCloudOriSurfFlag;

    map<int, pcl::PointCloud<PointType>> laserCloudMapContainer; //地图容器 ，first是索引，second是角点地图和平面地图

    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;// 从地图中提取的除当前帧外的当前帧的周围点云

    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;// 上面的降采样

//     pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
//     pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;


    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;

    
    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;
    nav_msgs::Odometry odomCur, odomLast;
    // float transformTobeMapped[6]; //当前的里程计，  RPYxyz初始化为0,0,0,0,0,0
    std::mutex mBuf;
    std::mutex mtx;
    
    std::mutex mtxLoopInfo;

//     bool isDegenerate = false;
//     Eigen::Matrix<float, 6, 6> matP;

//     int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
//     int laserCloudCornerLastDSNum = 0;
//     int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // 闭环容器，第一个保存新的闭环帧，第二个保存对应的旧闭环帧from new to old
//     // 用于gtsam优化的闭环关系队列
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::Float64MultiArray> loopInfoVec; // 闭环时间序列，每一个序列代表一次闭合，下标0存储当前闭环时间，1存储对应的之前的闭环时间

    nav_msgs::Path globalPath;

//     Eigen::Affine3f transPointAssociateToMap;// transformTobeMapped的矩阵形式
//     Eigen::Affine3f incrementalOdometryAffineFront; // save current transformation before any processing 相当于slam里程计值
//     Eigen::Affine3f incrementalOdometryAffineBack; //  经过scan2map优化后的值，又经过了imu差值后的值
    

//     // SC loop detector 
    SCManager scManager;
    int SCclosestHistoryFrameID; //  基于scan-context搜索的闭环关键帧
    float yawDiffRad;

    bool gps_initailized;
    bool pose_initailized;
    bool Calib_flag ;
    struct myPose{
        Eigen::Quaterniond Qwl=Eigen::Quaterniond::Identity();
        Eigen::Vector3d twl=Eigen::Vector3d::Zero() ;
        double timestamp=-1.0;
    }CurPose, deltaOdom, LastPose;
//     struct CalibrationExRotation{
//         Eigen::Matrix3d Rwl;
//         double timestamp;
//     };
//     queue<CalibrationExRotation> lidar_cali;
//     queue<CalibrationExRotation> gps_cali;
//     // Eigen::Matrix3d ric = Eigen::Matrix3d::Identity();
//     Eigen::Matrix3d ric;


    Eigen::Quaterniond initialQ = Eigen::Quaterniond::Identity();

    mapOptimization():gps_initailized(false),pose_initailized(false),Calib_flag(false),Qil(m_Ril)
    {

//         ric = extRot;

        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        // pubKeyPoses                 = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1);// 发布关键帧点云
        pubLaserCloudSurround       = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1);
        pubLaserOdometryGlobal      = m_ros_node_handle.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry", 1); // 全局里程计，有闭环优化
//         pubLaserOdometryIncremental = m_ros_node_handle.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry_incremental", 1); // 发布增量里程计，不受闭环优化等的影响，只用来算增量，是为了防止在算增量过程中发生优化，导致增量出错
        pubPath                     = m_ros_node_handle.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1); // 发布路径
        subOdom = m_ros_node_handle.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 5, &mapOptimization::OdomHandler, this, ros::TransportHints().tcpNoDelay());
        subCloud = m_ros_node_handle.subscribe<sensor_msgs::PointCloud2>("/surfcloud", 5, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subGPS   = m_ros_node_handle.subscribe<nav_msgs::Odometry> (gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
//         subLoop  = m_ros_node_handle.subscribe<std_msgs::Float64MultiArray>("lio_loop/to_be_added", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());

        pubHistoryKeyFrames   = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1); // 历史闭环帧附件的点云
        pubIcpKeyFrames       = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1); // 闭环矫正后的当前帧点云
        pubLoopConstraintEdge = m_ros_node_handle.advertise<visualization_msgs::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1); // 可视化闭环关系

//         pubRecentKeyFrames    = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
//         pubRecentKeyFrame     = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered", 1); // 当前帧的降采样点云
//         pubCloudRegisteredRaw = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);


        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
      

        allocateMemory();
        if(intialMethod=="human")
            gps_initailized=true;
        
        

       
    }

//     // 预先分配内存
    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

//         laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
//         laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
//         laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization
//         laserCloudInRaw.reset(new pcl::PointCloud<PointType>());
//         laserCloudInRaw->clear();

//         laserCloudOri.reset(new pcl::PointCloud<PointType>());
//         coeffSel.reset(new pcl::PointCloud<PointType>());

//         laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
//         coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
//         laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
//         laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
//         coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
//         laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

//         std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
//         std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);


        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());

        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

//         kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
//         kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

//         for (int i = 0; i < 6; ++i){
//             transformTobeMapped[i] = 0;
//         }

//         matP.setZero();
    }

// // 标定旋转外参
// bool Calibration_ExRotation(queue<CalibrationExRotation>& lidar_cali, // 上一帧和当前帧对应的特征点组
//                            queue<CalibrationExRotation>& gps_cali, // 两帧间预积分算出来的delta旋转四元数q
//                            Eigen::Matrix3d &calib_ric_result) // 标定出来的旋转矩阵
// {
 
//     if(lidar_cali.empty() || gps_cali.empty()){
//         cout<< "gps_cali为空"<<endl;
//         return false;
//     }
//     Eigen::Matrix3d ric(calib_ric_result);// ric初始值
//     queue<Eigen::Matrix3d> lidar_qu;
//     queue<Eigen::Matrix3d> gps_qu;

//     while(gps_cali.front().timestamp>lidar_cali.front().timestamp-0.01){
//         lidar_cali.pop();
//     }

//     while(!lidar_cali.empty()){
//         double lidar_time = lidar_cali.front().timestamp;
//         while(gps_cali.front().timestamp< lidar_time-0.01){
//             gps_cali.pop();
//         }
//         if(gps_cali.front().timestamp<lidar_time+0.01){
//             lidar_qu.push(lidar_cali.front().Rwl);
//             lidar_cali.pop();
//             gps_qu.push(gps_cali.front().Rwl);
//             gps_cali.pop();
//         }else{
//             lidar_cali.pop();
//             gps_cali.pop();
//         } 
//     }

//     cout<< lidar_qu.size() <<","<<gps_qu.size()<<endl;
//     if(lidar_qu.size() != gps_qu.size()){
      
//         cout << "长度不相等"<<endl;
//         return false;
//     }

//     queue<Eigen::Matrix3d> delta_lidar;
//     queue<Eigen::Matrix3d> delta_gps;
//     queue<Eigen::Matrix3d> Rc_g;
//     Eigen::Matrix3d last_lidar = lidar_qu.front();
//     lidar_qu.pop();
//     Eigen::Matrix3d last_gps = gps_qu.front();
//     gps_qu.pop();
//     // Ri,i+1 = Rwi.transpose() *  Rwi+1;
//     while(!lidar_qu.empty()){
//         delta_lidar.push(last_lidar.transpose() * lidar_qu.front());
//         last_lidar = lidar_qu.front();
//         lidar_qu.pop();

//         delta_gps.push(last_gps.transpose() * gps_qu.front());
//         last_gps = gps_qu.front();
//         gps_qu.pop();

//         Rc_g.push(ric.inverse() * delta_gps.front() * ric); 
//     }
//     // Sophus::SO3d SO3_R(delta_lidar.front());
//     // Vector3d so3 = SO3_R.log();
//     // if(so3.norm()<0.1){
//     //       cout << "旋转太小"<<endl;
//     //     return false;
      

//     // } 
    




  

//     Eigen::MatrixXd A(delta_lidar.size() * 4, 4);
//     A.setZero();
//     int sum_ok = 0; 
//     while(!delta_lidar.empty())
//     {
//         Eigen::Quaterniond r1(delta_lidar.front()); // 特征匹配得到的两帧间的旋转
//         Eigen::Quaterniond r2(Rc_g.front());// 预积分得到的两帧间的旋转

//         // https://www.iiiff.com/article/389681 可以理解为两个角之间的角度差
//         double angular_distance = 180 / M_PI * r1.angularDistance(r2);
//         cout << "角度差" << angular_distance<<endl;
        
//         // 一个简单的核函数
//         double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
//         ++sum_ok;
//         ROS_DEBUG(
//             "%d %f", sum_ok, angular_distance);
//         Eigen::Matrix4d L, R;
//         // 四元数的左乘矩阵
//         double w = Quaterniond(delta_lidar.front()).w();
//         Eigen::Vector3d q = Quaterniond(delta_lidar.front()).vec();
//         L.block<3, 3>(0, 0) = w * Eigen::Matrix3d::Identity() + gtsam::skewSymmetric(q);
//         L.block<3, 1>(0, 3) = q;
//         L.block<1, 3>(3, 0) = -q.transpose();
//         L(3, 3) = w;
//         // 四元数的右乘矩阵
//         Eigen::Quaterniond R_ij(delta_gps.front());
//         w = R_ij.w();
//         q = R_ij.vec();
//         R.block<3, 3>(0, 0) = w * Eigen::Matrix3d::Identity() - gtsam::skewSymmetric(q);
//         R.block<3, 1>(0, 3) = q;
//         R.block<1, 3>(3, 0) = -q.transpose();
//         R(3, 3) = w;

//         A.block<4, 4>((sum_ok - 1) * 4, 0) = huber * (L - R);    // 作用在残差上面
//         Rc_g.pop();
//         delta_lidar.pop();
//         delta_gps.pop();
//     }


//     Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
//     Eigen::Matrix<double, 4, 1> xxxx = svd.matrixV().col(3);
//     Eigen::Quaterniond estimated_R(xxxx);
//     ric = estimated_R.toRotationMatrix().inverse();
//     //cout << svd.singularValues().transpose() << endl;
//     //cout << ric << endl;
//     Eigen::Vector3d ric_cov;
//     ric_cov = svd.singularValues().tail<3>();
//     // 倒数第二个奇异值，因为旋转是3个自由度，因此检查一下第三小的奇异值是否足够大，通常需要足够的运动激励才能保证得到没有奇异的解
//     if ( ric_cov(1) > 0.25)
//     {
//         calib_ric_result = ric;
//         return true;
//     }
//     else{
//            cout<< "自由度不够:"<<ric_cov(1)<<endl;
//          cout << ric.transpose() << endl;
//                 Eigen::Vector3d eulerAngle=ric.transpose().eulerAngles(0,1,2);
//                 cout << "四元数" <<endl;
//                 cout << eulerAngle << endl;
//         return false;
     
//     }

    
        
// }



    int initial_count=0;


    void laserCloudInfoHandler(const sensor_msgs::PointCloud2::ConstPtr& msgIn)
    {
     
        if(!gps_initailized){
            cout<<"GPS not initailized"<<endl;
            return;
        }
        mBuf.lock();
        cloudQueue.push_back(*msgIn);
        mBuf.unlock();
    }

    void map_build(){
        ros::Rate rate( 1000 );
        myPose tmp_cur, tmp_last;
        while(ros::ok()){
        
            while(!cloudQueue.empty() && !odomQueue.empty()){
                mBuf.lock();  
                sensor_msgs::PointCloud2 cloudMsgIn = cloudQueue.front();
                cloudQueue.pop_front();
                timeLaserInfoCur = cloudMsgIn.header.stamp.toSec();
                timeLaserInfoStamp = cloudMsgIn.header.stamp;
                while (!odomQueue.empty() && odomQueue.front().header.stamp.toSec() < (cloudMsgIn.header.stamp.toSec()-0.01))
                    odomQueue.pop_front();

                     // 没有里程数据了,跳出
                if (odomQueue.empty())
                {
                    cout  << "odomQueue.empty()" <<endl;
                    mBuf.unlock();
                    break;
                }
                
                odomCur = odomQueue.front();
                odomQueue.pop_front();
                laserCloudSurfLast->clear();
                pcl::fromROSMsg(cloudMsgIn, *laserCloudSurfLast);
                mBuf.unlock();
                // cout << "cloudKeyPoses3D->size():"<<cloudKeyPoses3D->size()<<endl;
                LastPose = CurPose;
                tmp_last = tmp_cur;
                Eigen::Vector3d Pwi(odomCur.pose.pose.position.x,odomCur.pose.pose.position.y,odomCur.pose.pose.position.z);
                Eigen::Quaterniond Qwi(odomCur.pose.pose.orientation.w,odomCur.pose.pose.orientation.x,odomCur.pose.pose.orientation.y,odomCur.pose.pose.orientation.z);
                tmp_cur.twl= Pwi+ Qwi.matrix()*m_til;
                tmp_cur.Qwl = Qwi *Qil;
                if (cloudKeyPoses3D->points.empty()){
                    CurPose.twl.x() = initialPose.at(0);
                    CurPose.twl.y() = initialPose.at(1);
                    CurPose.twl.z() = initialPose.at(2);
                    CurPose.Qwl = initialQ;
                    CurPose.timestamp = timeLaserInfoCur;
                    // std::cout << " initialQ:" <<  initialQ.coeffs().transpose()<<std::endl;
                    // std::cout << " CurPose.Qwl:" <<  CurPose.Qwl.coeffs().transpose()<<std::endl;
                }
                else{
                    
                   
                   

                   
                    Eigen::Quaterniond last_Qlw =  tmp_last.Qwl.conjugate();
                    // T_i,i+1
                    deltaOdom.twl = last_Qlw*tmp_cur.twl-last_Qlw*tmp_last.twl;
                    deltaOdom.Qwl = last_Qlw*tmp_cur.Qwl;

                    CurPose.twl = LastPose.Qwl*deltaOdom.twl+LastPose.twl;
                    CurPose.Qwl = LastPose.Qwl*deltaOdom.Qwl;
                    CurPose.timestamp = timeLaserInfoCur;
                    // std::cout << " CurPose.Qwl:" <<  CurPose.Qwl.coeffs().transpose()<<std::endl;
                    
                }
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    saveKeyFramesAndFactor();  
                    correctPoses();
                }
                publishOdometry();

                if (pubPath.getNumSubscribers() != 0)
                {
                    globalPath.header.stamp = timeLaserInfoStamp;
                    globalPath.header.frame_id = "world";
                    pubPath.publish(globalPath);
                }

            }
            //t1.toc("建图用时");
            rate.sleep();
        }
    }

  
    void OdomHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
   
        // Eigen::Vector3d Pwl;
        // Eigen::Vector3d Pwi(odomMsg->pose.pose.position.x,odomMsg->pose.pose.position.y,odomMsg->pose.pose.position.z);
        // Eigen::Quaterniond Qwi(odomMsg->pose.pose.orientation.w,odomMsg->pose.pose.orientation.x,odomMsg->pose.pose.orientation.y,odomMsg->pose.pose.orientation.z);
        
        // Pwl= Pwi+ Qwi.matrix()*m_til;
        // Eigen::Quaterniond Qwl = Qwi *m_Ril;
        if(cloudKeyPoses3D->points.empty()){
      
            Eigen::Quaterniond Qwi(odomMsg->pose.pose.orientation.w,odomMsg->pose.pose.orientation.x,odomMsg->pose.pose.orientation.y,odomMsg->pose.pose.orientation.z);
            cout << "Qil:" << Qil.coeffs().transpose() <<endl;
            cout << "m_til" << m_til <<endl; 
            Eigen::Quaterniond Qwl = Qwi *Qil;
            initialQ = Qwl;
    
        }
  
        {
            mBuf.lock();
            odomQueue.push_back(*odomMsg);
            mBuf.unlock();
   
        }

            
        // cout<<"收到odom"<<endl;
    }


    // 添加GPS里程计数据到队列
    int gps_count=0;
    std::chrono::steady_clock::time_point now;
    std::chrono::steady_clock::time_point last;
    double last_E,last_N,last_U;
    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        // // 每隔一秒接收一次数据
        // ++gps_count;
        // gps_count%=200;
        // if(gps_count!=0){    
        //     return;
        // }
        // now = std::chrono::steady_clock::now();
        
        // double t_track = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(now - last).count();
        // last = now;
        // cout<<"一个周期的时间是"<<t_track<<endl;
        if(intialMethod=="gps"){
            if(!gps_initailized&&(gpsMsg->pose.pose.position.x!=0||gpsMsg->pose.pose.position.y!=0)&&(gpsMsg->pose.covariance[0]<0.003&&gpsMsg->pose.covariance[7]<0.003)){

                Eigen::Vector3d Pwl;
                Eigen::Vector3d Pwi(gpsMsg->pose.pose.position.x,gpsMsg->pose.pose.position.y,gpsMsg->pose.pose.position.z);
                Eigen::Quaterniond Qwi(gpsMsg->pose.pose.orientation.w,gpsMsg->pose.pose.orientation.x,gpsMsg->pose.pose.orientation.y,gpsMsg->pose.pose.orientation.z);
                Pwl= Pwi+ Qwi.matrix()*m_til;
                cout<<"GPS initailizes"<<endl;
                initialPose.at(0)=Pwl.x();
                initialPose.at(1)=Pwl.y();
              
                gps_initailized=true;
                last_E=initialPose.at(0);
                last_N=initialPose.at(1);
            }
        }
 
        if(optimization_with_GPS){
            if(last_E!=gpsMsg->pose.pose.position.x||last_N!=gpsMsg->pose.pose.position.y){
                mBuf.lock();
                gpsQueue.push_back(*gpsMsg);
                mBuf.unlock();    
            }
      
                
        }

            // 外参标定
        // if(Calib_flag)
        // {                
        //     CalibrationExRotation tmp;
        //     Eigen::Quaterniond Qwi(gpsMsg->pose.pose.orientation.w,gpsMsg->pose.pose.orientation.x,gpsMsg->pose.pose.orientation.y,gpsMsg->pose.pose.orientation.z);
        //     // Eigen::Vector3d Pwi(gpsMsg->pose.pose.position.x,gpsMsg->pose.pose.position.y,gpsMsg->pose.pose.position.z);
        //     // Pwl= Pwi+ Qwi.matrix()*Pil;
        //     tmp.Rwl = Qwi.toRotationMatrix();
        //     tmp.timestamp = gpsMsg->header.stamp.toSec();
        //     gps_cali.push(tmp);
        // }
            
        // cout<<"收到GPS"<<endl;
    }




       void pointTransForm(PointType const * const pi, PointType * const po,Eigen::Affine3f TransForm)
    {
        po->x = TransForm(0,0) * pi->x + TransForm(0,1) * pi->y + TransForm(0,2) * pi->z + TransForm(0,3);
        po->y = TransForm(1,0) * pi->x + TransForm(1,1) * pi->y + TransForm(1,2) * pi->z + TransForm(1,3);
        po->z = TransForm(2,0) * pi->x + TransForm(2,1) * pi->y + TransForm(2,2) * pi->z + TransForm(2,3);
        po->intensity = pi->intensity;
    }

    // 对输入点云进行位姿变换
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        // https://blog.csdn.net/bigFatCat_Tom/article/details/98493040
        // 使用多线程并行加速
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(myPose &CurPose_)
    {
        return gtsam::Pose3(gtsam::Rot3(CurPose_.Qwl), 
                                  gtsam::Point3(CurPose_.twl));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    // 从xyzRPY的单独数据变成仿射变换矩阵
    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    // 发布全局地图和保存地图
    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }

        // if(Calib_flag && lidar_cali.size()>100){
        //     cout << "开始标定" <<endl;
        //     if(Calibration_ExRotation(lidar_cali, gps_cali,ric)){
        //             cout << "外参标定成功" <<endl;
        //             Calib_flag = false;
        //     }                   
        //     else{
        //         cout << "外参标定失败" <<endl;
        //     }
        // }
        // if(!Calib_flag){
        //     cout << "外参标定" <<endl;
        //     cout << ric.transpose() << endl;
        //     Eigen::Vector3d eulerAngle=ric.transpose().eulerAngles(0,1,2);
        //     cout << "欧拉角" <<endl;
        //     cout << eulerAngle << endl;         
        // }

        if (savePCD == false)
            return;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        // create directory and remove old files;
        savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
        system((std::string("exec rm -r ") + savePCDDirectory).c_str());
        system((std::string("mkdir ") + savePCDDirectory).c_str());
        // save key frame transformations
        pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileASCII(savePCDDirectory + "transformations.pcd", *cloudKeyPoses6D);
        // extract global point cloud map        
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());

        // TODO 不是有个地图容器嘛，怎么还搞便利呀。不过这边建图都结束了倒也不在乎时间
        // 但是这样不会有很多重复的点嘛
        for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
      
            *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
        }
        // down-sample and save corner cloud
        // downSizeFilterCorner.setInputCloud(globalCornerCloud);
        // downSizeFilterCorner.filter(*globalCornerCloudDS);
        // pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloud);
        // down-sample and save surf cloud
        // downSizeFilterSurf.setInputCloud(globalSurfCloud);
        // downSizeFilterSurf.filter(*globalSurfCloudDS);
        // pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloud);
        // down-sample and save global point cloud map
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
    }

    void publishGlobalMap()
    {
     
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;
  
        if (cloudKeyPoses3D->points.empty() == true)
            return;
      
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(10.0, 10.0, 10.0); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
     
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        
        publishCloud<PointType>(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }

    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        {
            rate.sleep();
            performLoopClosure();
            visualizeLoopClosure();
        }
    }

//     // 暂时还没使用,接收人工闭环信息
//     void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
//     {
//         std::lock_guard<std::mutex> lock(mtxLoopInfo);
//         if (loopMsg->data.size() != 2)
//             return;

//         loopInfoVec.push_back(*loopMsg);

//         while (loopInfoVec.size() > 5)
//             loopInfoVec.pop_front();
//     }

    // 闭环检测当前帧和历史闭环关键帧
    // 找出对应的点云
    // 发布历史闭环关键帧点云
    // icp匹配
    // 将当前帧点云转到icp匹配位置，并发布
    // 存储闭环关系用于gtsam优化
    void performLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur; // 当前帧id
        int loopKeyPre; // 闭环帧id
        int SCloopKeyCur; // 当前帧id
        int SCloopKeyPre; // 闭环帧id
        // if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false){
        //     if (detectLoopClosureScanContext(&loopKeyCur, &loopKeyPre) == false){
        //         if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false){
        //             return;
        //         // }       
        //     }
        // }
        // if (detectLoopClosureScanContext(&SCloopKeyCur, &SCloopKeyPre) == false && detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false){
        //     return;
        // }       
        
        bool isValidRSloopFactor = false;
        bool isValidSCloopFactor = false;
        isValidRSloopFactor = detectLoopClosureDistance(&loopKeyCur, &loopKeyPre); // 开启rs
        // isValidSCloopFactor = detectLoopClosureScanContext(&SCloopKeyCur, &SCloopKeyPre); //开启sc
        // RS loop closure
        if (isValidRSloopFactor==false && isValidSCloopFactor==false ){
            return;
        }
        // RS loop closure
        if(isValidRSloopFactor){
            // extract cloud
            pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
            {
                loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
                loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
                if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                    return;
                if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                    publishCloud<PointType>(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
            }

            // ICP Settings
            static pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
            icp.setMaximumIterations(100);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setRANSACIterations(0);

            // Align clouds
            icp.setInputSource(cureKeyframeCloud);
            icp.setInputTarget(prevKeyframeCloud);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result);

            if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore){
                std::cout<<"ICP failed"<<std::endl;
                return;
            }
            // publish corrected cloud
            if (pubIcpKeyFrames.getNumSubscribers() != 0)
            {
                pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
                pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
                publishCloud<PointType>(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
            }

            // Get pose transformation
            float x, y, z, roll, pitch, yaw;
            Eigen::Affine3f correctionLidarFrame;
            correctionLidarFrame = icp.getFinalTransformation();
            // transform from world origin to wrong pose
            Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
            // transform from world origin to corrected pose
            Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
            pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
            gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
            gtsam::Vector Vector6(6);
            float noiseScore = icp.getFitnessScore();
            Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
            noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

            // Add pose constraint
            mtx.lock();
            loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
            loopPoseQueue.push_back(poseFrom.between(poseTo));
            loopNoiseQueue.push_back(constraintNoise);
            mtx.unlock();
            std::cout<<"RS 闭环成功"<<std::endl;
            // add loop constriant
            loopIndexContainer[loopKeyCur] = loopKeyPre;
        }
        // SC loop closure
        if(isValidSCloopFactor){
             // extract cloud
            pcl::PointCloud<PointType>::Ptr SCCurKeyFrameCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr SCCurKeyFrameCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr SCprevKeyframeCloud(new pcl::PointCloud<PointType>());

            {
            
                *SCCurKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[SCloopKeyCur],   &copy_cloudKeyPoses6D->points[SCloopKeyPre]);
                downSizeFilterICP.setInputCloud(SCCurKeyFrameCloud);
                downSizeFilterICP.filter(*SCCurKeyFrameCloudDS);
                loopFindNearKeyframes(SCprevKeyframeCloud,SCloopKeyPre,historyKeyframeSearchNum);
                if (SCCurKeyFrameCloudDS->size() < 300 || SCprevKeyframeCloud->size() < 1000)
                    return;
                if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                    publishCloud<PointType>(&pubHistoryKeyFrames, SCprevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
            }

            // ICP Settings
            static pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
            icp.setMaximumIterations(100);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setRANSACIterations(0);

            // Align clouds
            icp.setInputSource(SCCurKeyFrameCloudDS);
            icp.setInputTarget(SCprevKeyframeCloud);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result);

            if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore){
                std::cout<<"SC ICP failed"<<std::endl;
                return;
            }
            // publish corrected cloud
            if (pubIcpKeyFrames.getNumSubscribers() != 0)
            {
                pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
                pcl::transformPointCloud(*SCCurKeyFrameCloudDS, *closed_cloud, icp.getFinalTransformation());
                publishCloud<PointType>(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
            }

            // Get pose transformation
            float x, y, z, roll, pitch, yaw;
            Eigen::Affine3f correctionLidarFrame;
            correctionLidarFrame = icp.getFinalTransformation();
    
            pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
            gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));
            gtsam::Vector Vector6(6);
            float noiseScore = icp.getFitnessScore();
            Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
            noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

            // Add pose constraint
            mtx.lock();
            loopIndexQueue.push_back(make_pair(SCloopKeyCur, SCloopKeyPre));
            loopPoseQueue.push_back(poseFrom.between(poseTo));
            loopNoiseQueue.push_back(constraintNoise);
            mtx.unlock();
            std::cout<<"SC 闭环成功"<<std::endl;
            // add loop constriant
            loopIndexContainer[SCloopKeyCur] = SCloopKeyPre;
        }
    }

    // 距离闭环 // 成功返回true，失败返回false
    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        // 检测这个关键帧是否已经在闭环容器里了，防止重复闭环
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        
        // 在历史关键帧中找，时间相差超过阈值即可
        // TODO 这样岂不是原地不动也会闭环，应该还得关键帧序号相差超过一定阈值才行
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].timestamp - timeLaserInfoCur) > historyKeyframeSearchTimeDiff && loopKeyCur-id>10)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

 

        return true;
    }
    
    // ScanContext闭环 // 成功返回true，失败返回false
    bool detectLoopClosureScanContext(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        // 检测这个关键帧是否已经在闭环容器里了，防止重复闭环
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame    
        SCclosestHistoryFrameID=-1;   
        auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
        SCclosestHistoryFrameID = detectResult.first;
        yawDiffRad = detectResult.second; // 没有使用not use for v1 (because pcl icp withi initial somthing wrong...)
        // if all close, reject
        if (SCclosestHistoryFrameID == -1){ 
            return false;
        }

        // 在历史关键帧中找，时间相差超过阈值即可
        {
            int id = SCclosestHistoryFrameID;
            if (abs(copy_cloudKeyPoses6D->points[id].timestamp - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
            }
      
        }
        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    // loopInfoHandler 实现的人工闭环，暂时没有用
    // 成功返回true，失败返回false
    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();
        // 时间太近
        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
         // 找出离当前闭合时间最近的关键帧索引
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].timestamp > loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        // 找出离之前闭合时间最近的关键帧索引
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].timestamp < loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        //存在返回false，说明这个人工闭环已经被检测到过了，也就不需要了
        if (it != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }


    /**
     * @details 找出索引关键帧key前后searchNum范围内关键帧对应的点云
     * @param nearKeyframes 关键帧对应的搜索范围内的点云（包含角点云和面点云经过降采样）
     * @param key 关键帧索引
     * @param searchNum 前后搜索数量 2*searchNum
     */ 
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            // 转到世界坐标系下
       
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    // 可视化闭环关系，将闭环的两个帧连线
    void visualizeLoopClosure()
    {
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = "world";
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = "world";
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1; markerEdge.scale.y = 0.1; markerEdge.scale.z = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }



    // 被注释了，没有使用
    // 提取最近添加的n个关键帧
    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

   
        // 从地图中提取点云，如果地图容器中没有就加入进去
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
    
        laserCloudSurfFromMap->clear(); 
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            // 距离过大去掉
            // 10秒内的？？ 毕竟已经检验过一次距离了
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > 50.0)
                continue;

            // intensity 该帧对应的关键帧的序号，
            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            // 判断是否已经在laserCloudMapContainer容器中
            // 是，则直接从地图容器中提取周围点云
            // 否，则提取的同时加入到地图容器中
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
          
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd];
            } else {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] =laserCloudSurfTemp;
            }
            
        }


        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        // 防止地图缓存太大了
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

 





    

    // 最大最小值范围约束
    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    // 判断是否需要保存关键帧，
    // 当RPY角度或者位移大于阈值，则为true
    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        // 提取最后一个关键帧点云的周围50米范围内的关键帧
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        PointType tmp;
        tmp.x = CurPose.twl.x();
        tmp.y = CurPose.twl.y();
        tmp.z = CurPose.twl.z();
        kdtreeSurroundingKeyPoses->radiusSearch(tmp, (double)3, pointSearchInd, pointSearchSqDis);
        if(pointSearchInd.size()>30) return false; // 防止一个地方打转
        
        float x = deltaOdom.twl.x();
        float y = deltaOdom.twl.y();
        float z = deltaOdom.twl.z();
        Eigen::Vector3d eulerAngle=deltaOdom.Qwl.matrix().eulerAngles(0,1,2);

        if (abs(eulerAngle(0))  < surroundingkeyframeAddingAngleThreshold &&
            abs(eulerAngle(1)) < surroundingkeyframeAddingAngleThreshold && 
            abs(eulerAngle(2))   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    // 添加里程计因子
    void addOdomFactor()
    {
        // 为空添加先验因子
        if (cloudKeyPoses3D->points.empty())
        {
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(CurPose), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(CurPose));
        }else{
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            // gtsam::Pose3 poseFrom = trans2gtsamPose(LastPose);
            gtsam::Pose3 poseTo   = trans2gtsamPose(CurPose);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    // 对齐gps时间戳添加gps因子
    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty()) // 没有关键帧，不添加gps
            return;
        else
        {
            // 关键帧距离太近，不添加
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        // 位置协方差太小，不添加
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        // 对齐时间戳
        while (!gpsQueue.empty())
        {
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();
                Eigen::Vector3d Pwl;
                Eigen::Vector3d Pwi(thisGPS.pose.pose.position.x,thisGPS.pose.pose.position.y,thisGPS.pose.pose.position.z);
                Eigen::Quaterniond Qwi(thisGPS.pose.pose.orientation.w,thisGPS.pose.pose.orientation.x,thisGPS.pose.pose.orientation.y,thisGPS.pose.pose.orientation.z);
                Pwl= Pwi+ Qwi*m_til;
                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                float gps_x = Pwl.x();
                float gps_y = Pwl.y();
                float gps_z = Pwl.z();
                if (!useGpsElevation)
                {
                    gps_z = CurPose.twl.z();
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                // GPS未正确初始化（0,0,0）
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                //每隔几米增加一次GPS
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < gpsAddDis)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 0.1f), max(noise_y, 0.1f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;
                break;
            }
        }
    }

    // 添加闭环因子
    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    // 添加里程计、gps、闭环因子，并执行gtsam优化，保存优化后的当前关键帧
    // 保存关键帧点云
    // 发布关键帧路径
    void saveKeyFramesAndFactor()
    {
        // cout << "111" <<endl;
        if (saveFrame() == false)
            return;
        // cout << "222" <<endl;
        // // odom factor
        addOdomFactor();
       

        // cout << "333" <<endl;
        // // gps factor
        // addGPSFactor();
        // cout << "444" <<endl;
        // // loop factor
        // addLoopFactor();

     

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
      
        isam->update();
      
        // 闭环成功则更新
        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }
     
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        // 添加当前帧
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.timestamp = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        CurPose.twl = latestEstimate.translation();
        CurPose.Qwl = Eigen::Quaterniond(Eigen::Matrix3d(latestEstimate.rotation().matrix()));
        // CurPose.twl.x() = thisPose6D.x;
        // CurPose.twl.y() = thisPose6D.y;
        // CurPose.twl.z() = thisPose6D.z;
        // Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(thisPose6D.roll ,Eigen::Vector3d::UnitX()));
        // Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(thisPose6D.pitch,Eigen::Vector3d::UnitY()));
        // Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(thisPose6D.yaw,Eigen::Vector3d::UnitZ()));
        // CurPose.Qwl=yawAngle*pitchAngle*rollAngle;
       

  
        // poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);
      
        // // save updated transform
        // // transformTobeMapped[0] = latestEstimate.rotation().roll();
        // // transformTobeMapped[1] = latestEstimate.rotation().pitch();
        // // transformTobeMapped[2] = latestEstimate.rotation().yaw();
        // // transformTobeMapped[3] = latestEstimate.translation().x();
        // // transformTobeMapped[4] = latestEstimate.translation().y();
        // // transformTobeMapped[5] = latestEstimate.translation().z();
        

    


        // save all the received edge and surf points
        // pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        // pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLast,    *thisSurfKeyFrame);
        

        // // TODO 这里用的是特征点云，还可用原始点云
        // scManager.makeAndSaveScancontextAndKeys(*thisCloudKeyFrame);
        scManager.makeAndSaveScancontextAndKeys(*laserCloudSurfLast);

        // save key frame cloud

        surfCloudKeyFrames.push_back(thisSurfKeyFrame);
       
        // save path for visualization
        // 发布关键帧路径
        updatePath(thisPose6D);
       
    }

    // 当发生闭环时，更新所有关键帧的位姿和路径
    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear map cache
            laserCloudMapContainer.clear();
            // clear path
            globalPath.poses.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false;
        }
    }

    // 更新关键帧路径信息
    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.timestamp);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    // 发布优化真实的里程计和不经过优化的增量里程计
    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "lidar";
        laserOdometryROS.pose.pose.position.x = CurPose.twl.x();
        laserOdometryROS.pose.pose.position.y = CurPose.twl.y();
        laserOdometryROS.pose.pose.position.z = CurPose.twl.z();
        laserOdometryROS.pose.pose.orientation.x = CurPose.Qwl.x();
        laserOdometryROS.pose.pose.orientation.y = CurPose.Qwl.y();
        laserOdometryROS.pose.pose.orientation.z = CurPose.Qwl.z();
        laserOdometryROS.pose.pose.orientation.w = CurPose.Qwl.w();

        // std::cout << " CurPose.twl:" <<  CurPose.twl.transpose() <<std::endl;
        // std::cout << " CurPose.Qwl:" <<  CurPose.Qwl.coeffs().transpose()<<std::endl;

        pubLaserOdometryGlobal.publish(laserOdometryROS);
        // cout<< "普通xyz："<<transformTobeMapped[3]<<","<<transformTobeMapped[4]<<","<<transformTobeMapped[5]<<endl;
        // Publish TF
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::Quaternion(CurPose.Qwl.x(),CurPose.Qwl.y(),CurPose.Qwl.z(),CurPose.Qwl.w()),
                                                      tf::Vector3(CurPose.twl.x(), CurPose.twl.y(), CurPose.twl.z()));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar");
        br.sendTransform(trans_odom_to_lidar);

     
    }


};


int main(int argc, char** argv)
{
    Eigen::initParallel(); // 只要加入Eigen::initParallel()，Eigen就可以利用OpenMP的API来进行多核计算。
    ros::init(argc, argv, "lio_sam");
    
    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");
    
    // std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);
    std::thread buildthread(&mapOptimization::map_build, &MO);
    // ros::MultiThreadedSpinner spinner(3);
    // spinner.spin();
    ros::spin();

    // loopthread.join();
    visualizeMapThread.join();
    buildthread.join();

    return 0;
}
