#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <eigen3/Eigen/Dense>

#include <ceres/ceres.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>


#include "tictoc.h"
#include "utility.h"
#include "Scancontext.h"

using namespace gtsam;

using std::cout;
using std::endl;

#define dowmSample 0
#define G 9.80665
#define ellipse_a 6378137
#define ellipse_e 0.081819190842622
class Parse_GPS{
 public: 
  double ConvertMatrix[4][3];

  double base_point_lat;
  double base_point_lon;
  double base_point_hei;
  Parse_GPS()
  {
    base_point_lat=34.14860588;
    base_point_lon=108.58563350;
    base_point_hei=414.105000;

    // 计算转换矩阵
    GenerateConvertMatrix(ConvertMatrix,base_point_lat,base_point_lon,base_point_hei);
  }


  void GenerateConvertMatrix(double ConvertMatrix[4][3], double latitude, double longitude, double height){
    double N;

    latitude = latitude / 180.0 *M_PI ;
    longitude = longitude / 180.0 * M_PI;
    double sin_long=sin(longitude);
    double cos_long=cos(longitude);
    double sin_lat =sin(latitude);
    double cos_lat =cos(latitude);

    N = ellipse_a / (sqrt(1 - ellipse_e*ellipse_e*sin_lat*sin_lat));

    ConvertMatrix[1][0] = -sin_long;
    ConvertMatrix[1][1] = cos_long;
    ConvertMatrix[1][2] = 0;
    ConvertMatrix[0][0] = -sin_lat*cos_long;
    ConvertMatrix[0][1] = -sin_lat*sin_long;
    ConvertMatrix[0][2] = cos_lat;
    ConvertMatrix[2][0] = cos_lat*cos_long;
    ConvertMatrix[2][1] = cos_lat*sin_long;
    ConvertMatrix[2][2] = sin_lat;

    ConvertMatrix[3][0] = (N + height)*cos_lat*cos_long;
    ConvertMatrix[3][1] = (N + height)*cos_lat*sin_long;
    ConvertMatrix[3][2] = (N*(1 - ellipse_e*ellipse_e) + height)*sin_lat;
  }

  void PointData2ENU(double latitude, double longitude, double height, double &outputx, double &outputy, double &outputz)
  {
    
    double N, x1, y1, z1, dx, dy, dz, x0, y0, z0;
    x0 = ConvertMatrix[3][0]; y0 = ConvertMatrix[3][1]; z0 = ConvertMatrix[3][2];

    latitude = latitude / 180.0 * M_PI;
    longitude = longitude / 180.0 * M_PI;

    double sin_long=sin(longitude);
      double cos_long=cos(longitude);
      double sin_lat =sin(latitude);
      double cos_lat =cos(latitude);

    N = ellipse_a / (sqrt(1 - ellipse_e*ellipse_e*sin_lat*sin_lat));
    x1 = (N + height)*cos_lat*cos_long;
    y1 = (N + height)*cos_lat*sin_long;
    z1 = (N*(1 - ellipse_e*ellipse_e) + height)*sin_lat;
    dx = x1 - x0; dy = y1 - y0; dz = z1 - z0;
    outputy = ConvertMatrix[0][0] * dx + ConvertMatrix[0][1] * dy + ConvertMatrix[0][2] * dz;
    outputx = ConvertMatrix[1][0] * dx + ConvertMatrix[1][1] * dy;
    outputz = ConvertMatrix[2][0] * dx + ConvertMatrix[2][1] * dy + ConvertMatrix[2][2] * dz;

  }
};





typedef pcl::PointXYZI PointType;

inline double rad2deg(double radians)
{
  return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
  return degrees * M_PI / 180.0;
}

struct Pose6D {
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

Eigen::Affine3f PoseToAffine3f(Pose6D curPose_)
{ 
    return pcl::getTransformation(curPose_.x, curPose_.y, curPose_.z, curPose_.roll, curPose_.pitch, curPose_.yaw);
}

class Map_Build : public ParamServer
{

public:
Parse_GPS parseGPS;

bool aLoopIsClosed = false;

// 用于gtsam优化的闭环关系队列
vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;



double keyframeMeterGap;

double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0; // large value means must add the first given frame. 确保第一帧一定被添加
double rotaionAccumulated = 1000000.0; // large value means must add the first given frame.

bool isNowKeyFrame = false; 

Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init 
Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init pose is zero 

Eigen::Vector3d initialPose;
Eigen::Quaterniond initialQ;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;  // 里程计isam
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf; // 点云
std::queue<std::pair<Eigen::Vector3d,double>> gpsBuf;
std::queue<Eigen::Quaterniond> gpsBufQ;
// std::queue<sensor_msgs::NavSatFix::ConstPtr> gpsBuf; // gps

std::map<int, int> loopIndexContainer; // 闭环容器，第一个保存新的闭环帧，第二个保存对应的旧闭环帧from new to old

std::mutex mBuf;
std::mutex mKF;

double timeLaserOdometry = 0.0;
double timeLaser = 0.0;

pcl::PointCloud<PointType>::Ptr laserCloudFullRes;
pcl::PointCloud<PointType>::Ptr laserCloudMapAfterPGO;

std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds;  // 关键帧
std::vector<Pose6D> keyframePoses; // 关键帧里程计pose
std::vector<Pose6D> keyframePosesUpdated; // 优化更新后的pose
pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D; //gtsam优化后的地图关键帧位置(x，y，z,time)
pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;
std::vector<double> keyframeTimes; // 关键帧时间
int recentIdxUpdated = 0; // 上一次更新的ID

gtsam::NonlinearFactorGraph gtSAMgraph;
bool gtSAMgraphMade = false; // 第一帧
gtsam::Values initialEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;

noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Base::shared_ptr robustLoopNoise;
noiseModel::Base::shared_ptr robustGPSNoise;

pcl::VoxelGrid<PointType> downSizeFilterScancontext;
SCManager scManager;
double scDistThres, scMaximumRadius;

pcl::VoxelGrid<PointType> downSizeFilterICP;
std::mutex mtxICP;

std::mutex mtxRecentPose;

pcl::PointCloud<PointType>::Ptr laserCloudMapPGO;
pcl::VoxelGrid<PointType> downSizeFilterMapPGO;



Eigen::Vector3d currGPS;
Eigen::Quaterniond currGPSQ;
bool hasGPSforThisKF = false;
bool gps_initailized = false;
double gpsAltitudeInitOffset = 0.0;   // 第一帧GPS的yaw轴朝向
double recentOptimizedX = 0.0;  // 上一帧优化后的xy
double recentOptimizedY = 0.0;
double recentOptimizedZ = 0.0;
ros::Publisher pubMapAftPGO, pubOdomAftPGO, pubPathAftPGO;
ros::Publisher pubLoopScanLocal, pubLoopSubmapLocal;
ros::Publisher pubOdomRepubVerifier;
ros::Publisher pubLoopConstraintEdge;

ros::Subscriber subLaserCloudFullRes;
ros::Subscriber subLaserOdometry;
ros::Subscriber subGPS;
ros::Subscriber subGPS_Odom;

std::string save_directory;
std::string pgKITTIformat, pgScansDirectory;
std::string odomKITTIformat;
std::fstream groud_truth_stream;
std::string tum_groud_truth;
std::fstream pgTimeSaveStream;



std::string padZeros(int val, int num_digits = 6) {
  std::ostringstream out;
  out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
  return out.str();
}

Map_Build(){

    initialPose = Eigen::Vector3d::Zero();
    initialQ = Eigen::Quaterniond::Identity();
    currGPSQ = Eigen::Quaterniond::Identity();
    // 保存轨迹
	m_ros_node_handle.param<std::string>("save_directory", save_directory, "/"); // pose assignment every k m move 
    auto unused = system((std::string("exec rm -r ") + save_directory).c_str());
    unused = system((std::string("mkdir -p ") + save_directory).c_str());
    unused = system((std::string("mkdir -p ") + save_directory+"optimaztion_data").c_str());
    unused = system((std::string("mkdir -p ") + save_directory+"odom_data").c_str());
    unused = system((std::string("mkdir -p ") + save_directory+"ground_truth_pose").c_str());
    pgKITTIformat = save_directory + "optimaztion_data/01_pred.txt";  // optimized_poses.txt
    odomKITTIformat = save_directory + "odom_data/01_pred.txt";
    tum_groud_truth=save_directory + "ground_truth_pose/01.txt";
    groud_truth_stream.open(tum_groud_truth.c_str(), fstream::out);
    groud_truth_stream << fixed;
    pgTimeSaveStream = std::fstream(save_directory + "times.txt", std::fstream::out); 
    pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);
    pgScansDirectory = save_directory + "Scans/";
    unused = system((std::string("exec rm -r ") + pgScansDirectory).c_str());
    unused = system((std::string("mkdir -p ") + pgScansDirectory).c_str());



    keyframeRadGap = deg2rad(keyframeDegGap);

	m_ros_node_handle.param<double>("sc_dist_thres", scDistThres, 0.3);  
	m_ros_node_handle.param<double>("sc_max_radius", scMaximumRadius, 80.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor 

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);
    initNoises();

    if(intialMethod=="human"){
        initialPose.x()= initial_Pose[0];
        initialPose.y()= initial_Pose[1];
        initialPose.z()= initial_Pose[2];
        gps_initailized=true;
    }
        

    //  初始化
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    laserCloudFullRes.reset(new pcl::PointCloud<PointType>());
    laserCloudMapPGO.reset(new pcl::PointCloud<PointType>());
    laserCloudMapAfterPGO.reset(new pcl::PointCloud<PointType>());


    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);

    float filter_size = 0.4; 
    downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);
    downSizeFilterICP.setLeafSize(filter_size, filter_size, filter_size);

    double mapVizFilterSize;
	m_ros_node_handle.param<double>("mapviz_filter_size", mapVizFilterSize, 0.05); // pose assignment every k frames 
    downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);

	subLaserCloudFullRes = m_ros_node_handle.subscribe<sensor_msgs::PointCloud2>("/surfcloud", 100, &Map_Build::laserCloudFullResHandler,this,ros::TransportHints().tcpNoDelay());
	subLaserOdometry = m_ros_node_handle.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100,  &Map_Build::laserOdometryHandler,this,ros::TransportHints().tcpNoDelay());
	subGPS = m_ros_node_handle.subscribe<sensor_msgs::NavSatFix>(gpsTopic, 100,  &Map_Build::gpsHandler,this,ros::TransportHints().tcpNoDelay());
    subGPS_Odom = m_ros_node_handle.subscribe<nav_msgs::Odometry>(gpsOdomTopic, 100,  &Map_Build::gpsOdomHandler,this,ros::TransportHints().tcpNoDelay());

	pubOdomAftPGO = m_ros_node_handle.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
	pubOdomRepubVerifier = m_ros_node_handle.advertise<nav_msgs::Odometry>("/repub_odom", 100);
	pubPathAftPGO = m_ros_node_handle.advertise<nav_msgs::Path>("/aft_pgo_path", 100);
	pubMapAftPGO = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/aft_pgo_map", 100);

	pubLoopScanLocal = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/loop_scan_local", 100);
	pubLoopSubmapLocal = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/loop_submap_local", 100);
    pubLoopConstraintEdge = m_ros_node_handle.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints", 1); // 可视化闭环关系
}


gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6DtoGTSAMPose3

void saveOdometryVerticesTUMformat(std::string _filename)
{
    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);
    stream << fixed;
    for(int i=0; i<keyframePoses.size();++i ) {
        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(keyframePoses[i]);
        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3
        Eigen::Matrix3d Rot;
        Rot<< col1.x(),col2.x(),col3.x(),
            col1.y(),col2.y(),col3.y(),
            col1.z(),col2.z(),col3.z();
        Eigen::Quaterniond q(Rot);

        stream << setprecision(6) << keyframeTimes[i]<< setprecision(7) << " " << t.x() << " "<< t.y() << " " << t.z()<< " " << q.x() << " " << q.y()<< " " << q.z() << " "<< q.w() << endl;
 
    }
}

void saveOptimizedVerticesTUMformat(gtsam::Values _estimates, std::string _filename)
{
    using namespace gtsam;

    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);
    stream << fixed;
    for(int i=0; i<keyframePosesUpdated.size();++i ) {
        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(keyframePosesUpdated[i]);

       

        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3
        Eigen::Matrix3d Rot;
        Rot<<  col1.x(),col2.x(),col3.x(),
                 col1.y(),col2.y(),col3.y(),
                 col1.z(),col2.z(),col3.z();
        Eigen::Quaterniond q(Rot);

        stream << setprecision(6) << keyframeTimes[i] << setprecision(7) << " " << t.x() << " "<< t.y() << " " << t.z()<< " " << q.x() << " " << q.y()<< " " << q.z() << " "<< q.w() << endl;
        // stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
        //        << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
        //        << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

void saveOdometryVerticesKittiformat(std::string _filename)
{
    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);
    stream << fixed;
    for(int i=0; i<keyframePoses.size();++i ) {
        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(keyframePoses[i]);
        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

void saveOptimizedVerticesKittiformat(gtsam::Values _estimates, std::string _filename)
{
    using namespace gtsam;

    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);
    stream << fixed;
    for(int i=0; i<keyframePosesUpdated.size();++i ) {
        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(keyframePosesUpdated[i]);

       

        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3
      

        // stream << setprecision(6) << keyframeTimes[i] << setprecision(7) << " " << t.x() << " "<< t.y() << " " << t.z()<< " " << q.x() << " " << q.y()<< " " << q.z() << " "<< q.w() << endl;
        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}


// 里程计
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
{
    
    if(!gps_initailized) return;
	mBuf.lock();
	odometryBuf.push(_laserOdometry);
	mBuf.unlock();
} // laserOdometryHandler

// 点云
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &_laserCloudFullRes)
{
   
    // cout << "intialMethod:" << intialMethod<<endl;
    // cout << "m_til:"<< m_til.transpose() <<endl;
    if(!gps_initailized){
        cout << "GPS not intialized:" << intialMethod<<endl;
        return;
    }
	mBuf.lock();
	fullResBuf.push(_laserCloudFullRes);
	mBuf.unlock();
} // laserCloudFullResHandler



// gps
void gpsHandler(const sensor_msgs::NavSatFix::ConstPtr &gpsMsg)
{
    Eigen::Quaterniond Qwi(0.000084,0.912099,-0.409690,-0.015147);
    Qwi.normalize();
    Qwi.conjugate();
    if(intialMethod=="gps"){
        if(!gps_initailized&&(gpsMsg->latitude!=0||gpsMsg->longitude!=0)&&(gpsMsg->position_covariance[0]<0.1&&gpsMsg->position_covariance[4]<0.1)){
            double tmp_x,tmp_y,tmp_z;
            parseGPS.PointData2ENU(gpsMsg->latitude/100.0,gpsMsg->longitude/100.0,gpsMsg->altitude,tmp_x,tmp_y,tmp_z );
           
            cout<<"GPS initailizes"<<endl;
            initialPose.x()=tmp_x;
            initialPose.y()=tmp_y;
            initialPose.z() = tmp_z;
            initialQ = Qwi;
            gps_initailized=true;
        }
    }
    if(optimization_with_GPS&&(gpsMsg->position_covariance[0]<0.1&&gpsMsg->position_covariance[4]<0.1)){
        double tmp_x,tmp_y,tmp_z;
        parseGPS.PointData2ENU(gpsMsg->latitude/100.0,gpsMsg->longitude/100.0,gpsMsg->altitude,tmp_x,tmp_y,tmp_z );
        mBuf.lock();
        gpsBuf.push(make_pair(Eigen::Vector3d(tmp_x,tmp_y,recentOptimizedZ),gpsMsg->header.stamp.toSec()));
        mBuf.unlock();
    }

    
 
} // gpsHandler


void gpsOdomHandler(const nav_msgs::Odometry::ConstPtr &gpsMsg)
{
    Eigen::Vector3d Pwl;
    Eigen::Vector3d Pwi(gpsMsg->pose.pose.position.x,gpsMsg->pose.pose.position.y,gpsMsg->pose.pose.position.z);
    Eigen::Quaterniond Qwi(gpsMsg->pose.pose.orientation.w,gpsMsg->pose.pose.orientation.x,gpsMsg->pose.pose.orientation.y,gpsMsg->pose.pose.orientation.z);
    Qwi.normalize();

    Pwl= Pwi+ Qwi.matrix()*m_til;
   
    // cout << "m_til:"<< m_til.transpose() <<endl;
    // cout << "Pwi:"<< Pwi.transpose() <<endl;
    // cout << "Pwl:"<< Pwl.transpose() <<endl;
    // //  cout << "Qwi:"<< Qwi.coeffs() <<endl;
    // cout << "initialPose:"<< initialPose.transpose() <<endl;
    // Pwl.z()= recentOptimizedZ;
    Eigen::Quaterniond Qwl(Qwi.matrix()*m_Ril);
    if(intialMethod=="gps"){
        if(!gps_initailized&&(gpsMsg->pose.covariance[0]<0.1&&gpsMsg->pose.covariance[7]<0.1)){
            
            
           
            cout<<"GPS initailizes"<<endl;
            initialPose.x()=Pwi.x();
            initialPose.y()=Pwi.y();
            initialPose.z() =0;
            initialQ = Qwi;
            gps_initailized=true;
        }
    }
    if(optimization_with_GPS&&(gpsMsg->pose.covariance[0]<0.1&&gpsMsg->pose.covariance[7]<0.1)){
       
        
        mBuf.lock();
        gpsBuf.push(make_pair(Pwl,gpsMsg->header.stamp.toSec()));
        gpsBufQ.push(Qwl);
        mBuf.unlock();  
    }

    if(evalFormat=="tum"){
        
        groud_truth_stream << setprecision(6) << gpsMsg->header.stamp.toSec() << setprecision(7) << " " << Pwl.x() << " "<< Pwl.y() << " " << Pwl.z()<< " " << Qwl.x() << " " << Qwl.y()<< " " << Qwl.z() << " "<< Qwl.w() << endl;
    }
    

} // gpsHandler



void initNoises( void )
{
    gtsam::Vector priorNoiseVector6(6);
    priorNoiseVector6 << 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-8; // RPYxyz
    // priorNoiseVector6 << 10, 10, 10, 10, 10, 10; // RPYxyz
    priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

    gtsam::Vector odomNoiseVector6(6);
    // odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    double loopNoiseScore = 0.5; // constant is ok...
    gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );


    gtsam::Vector robustNoiseVector3(3); // gps factor has 3 elements (xyz)
    robustNoiseVector3 << 0.1, 0.1, 0.1; // means only caring altitude here. (because LOAM-like-methods tends to be asymptotically flyging)
    robustGPSNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector3) );

} // initNoises

// ROS的odom数据转 xyz rpy
Pose6D getOdom(nav_msgs::Odometry::ConstPtr _odom)
{

         
    Eigen::Vector3d tmpToi(_odom->pose.pose.position.x,_odom->pose.pose.position.y,_odom->pose.pose.position.z);
    Eigen::Quaterniond tmpQoi( _odom->pose.pose.orientation.w, _odom->pose.pose.orientation.x, _odom->pose.pose.orientation.y, _odom->pose.pose.orientation.z);

   
    Eigen::Quaterniond tmpQwl(initialQ.matrix()* tmpQoi.matrix()*m_Ril); 
    // cout << "tmpQwl before:" << tmpQwl.coeffs()<<endl;

    Eigen::Quaterniond tmpQwl_gps(currGPSQ);
    if(1){
        tmpQwl = tmpQwl.slerp(gpsPoseProportion,tmpQwl_gps);
    }
    
   

    Eigen::Vector3d tmpTwl= initialQ* tmpQoi*m_til+ initialQ*tmpToi+initialPose;



    

    double roll, pitch, yaw;

    tf::Matrix3x3(tf::Quaternion(tmpQwl.x(), tmpQwl.y(), tmpQwl.z(), tmpQwl.w())).getRPY(roll, pitch, yaw);

    return Pose6D{tmpTwl.x(), tmpTwl.y(), tmpTwl.z(), roll, pitch, yaw}; 
} // getOdom

// 两帧间的相对xyzrpy
Pose6D diffTransformation(const Pose6D& _p1, const Pose6D& _p2)
{
    Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
    Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
    Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
    Eigen::Affine3f SE3_delta; SE3_delta.matrix() = SE3_delta0;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
    // std::cout << "delta : " << dx << ", " << dy << ", " << dz << ", " << droll << ", " << dpitch << ", " << dyaw << std::endl;

    return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)), double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
} // SE3Diff

// 坐标变换
pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);
    
    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}

void pubPath( void )
{
    // pub odom and path 
    nav_msgs::Odometry odomAftPGO;
    nav_msgs::Path pathAftPGO;
    pathAftPGO.header.frame_id = "world";
    mKF.lock(); 
    // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()) - 1; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    {
        const Pose6D& pose_est = keyframePosesUpdated.at(node_idx); // upodated poses
        // const gtsam::Pose3& pose_est = isamCurrentEstimate.at<gtsam::Pose3>(node_idx);

        nav_msgs::Odometry odomAftPGOthis;
        odomAftPGOthis.header.frame_id = "world";
        odomAftPGOthis.child_frame_id = "/aft_pgo";
        odomAftPGOthis.header.stamp = ros::Time().fromSec(keyframeTimes.at(node_idx));
        odomAftPGOthis.pose.pose.position.x = pose_est.x;
        odomAftPGOthis.pose.pose.position.y = pose_est.y;
        odomAftPGOthis.pose.pose.position.z = pose_est.z;
        odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);
        odomAftPGO = odomAftPGOthis;

        geometry_msgs::PoseStamped poseStampAftPGO;
        poseStampAftPGO.header = odomAftPGOthis.header;
        poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

        pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
        pathAftPGO.header.frame_id = "world";
        pathAftPGO.poses.push_back(poseStampAftPGO);
    }
    mKF.unlock(); 
    pubOdomAftPGO.publish(odomAftPGO); // last pose 
    pubPathAftPGO.publish(pathAftPGO); // poses 

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x, odomAftPGO.pose.pose.position.y, odomAftPGO.pose.pose.position.z));
    q.setW(odomAftPGO.pose.pose.orientation.w);
    q.setX(odomAftPGO.pose.pose.orientation.x);
    q.setY(odomAftPGO.pose.pose.orientation.y);
    q.setZ(odomAftPGO.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftPGO.header.stamp, "world", "/aft_pgo"));
} // pubPath

void updatePoses(void)
{
    mKF.lock(); 
    for (int node_idx=0; node_idx < int(isamCurrentEstimate.size()); node_idx++)
    {
        Pose6D& p =keyframePosesUpdated[node_idx];
        p.x = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().x();
        p.y = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().y();
        p.z = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().z();
        p.roll = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().roll();
        p.pitch = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().pitch();
        p.yaw = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().yaw();

        cloudKeyPoses3D->points[node_idx].x = p.x;
        cloudKeyPoses3D->points[node_idx].y = p.y;
        cloudKeyPoses3D->points[node_idx].z = p.z;
    }
    mKF.unlock();

    mtxRecentPose.lock();
    const gtsam::Pose3& lastOptimizedPose = isamCurrentEstimate.at<gtsam::Pose3>(int(isamCurrentEstimate.size())-1);
    recentOptimizedX = lastOptimizedPose.translation().x();
    recentOptimizedY = lastOptimizedPose.translation().y();
    recentOptimizedZ = lastOptimizedPose.translation().z();
    recentIdxUpdated = int(keyframePosesUpdated.size()) - 1;

    mtxRecentPose.unlock();
} // updatePoses

void runISAM2opt(void)
{
    // called when a variable added 
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    updatePoses();
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, gtsam::Pose3 transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType *pointFrom;

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(
                                    transformIn.translation().x(), transformIn.translation().y(), transformIn.translation().z(), 
                                    transformIn.rotation().roll(), transformIn.rotation().pitch(), transformIn.rotation().yaw() );
    
    int numberOfCores = 8; // TODO move to yaml 
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
} // transformPointCloud

/**
 * @brief 找到当前帧周围的子图
 * 
 * @param nearKeyframes  当前帧周围的子图变换到目标帧坐标系下
 * @param key    当前帧ID
 * @param submap_size 自地图关键帧数量
 * @param root_idx 目标帧ID.没用
 */
void loopFindNearKeyframesCloud( pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& submap_size, const int& root_idx)
{
    // extract and stacking near keyframes (in global coord)
    nearKeyframes->clear();
    // 遍历当前帧ID前后的一些帧
    for (int i = -submap_size; i <= submap_size; ++i) {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= int(keyframeLaserClouds.size()) )
            continue;

        // 变换到目标帧的位姿下
        mKF.lock(); 
       
        // *nearKeyframes += * local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[root_idx]);   //! bug 
        *nearKeyframes += * local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[keyNear]); 
        mKF.unlock(); 
    }

    if (nearKeyframes->empty())
        return;
#if dowmSample
    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());

    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
#endif
} // loopFindNearKeyframesCloud

/**
 * @brief ICP验证闭环,闭环成功则返回闭环因子
 * 
 * @param _loop_kf_idx 历史帧ID
 * @param _curr_kf_idx 当前帧ID
 * @return std::optional<gtsam::Pose3> 
 */
std::optional<gtsam::Pose3> doICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx, float& score )
{
    // parse pointclouds
    int historyKeyframeSearchNum = 25; // enough. ex. [-25, 25] covers submap length of 50x1 = 50m if every kf gap is 1m
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());

    // 找到当前帧和历史帧的子图点云
    loopFindNearKeyframesCloud(cureKeyframeCloud, _curr_kf_idx, 0, _loop_kf_idx); // use same root of loop kf idx 
    loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum, _loop_kf_idx); 
    if(0){
        // loop verification 
        // 发布闭环点云
        sensor_msgs::PointCloud2 cureKeyframeCloudMsg;
        pcl::toROSMsg(*cureKeyframeCloud, cureKeyframeCloudMsg);
        cureKeyframeCloudMsg.header.frame_id = "world";
        pubLoopScanLocal.publish(cureKeyframeCloudMsg);

        sensor_msgs::PointCloud2 targetKeyframeCloudMsg;
        pcl::toROSMsg(*targetKeyframeCloud, targetKeyframeCloudMsg);
        targetKeyframeCloudMsg.header.frame_id = "world";
        pubLoopSubmapLocal.publish(targetKeyframeCloudMsg);
    }


    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter 
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align pointclouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(targetKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);
 
  
    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore) {
        std::cout << "[SC loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << historyKeyframeFitnessScore << "). Reject this SC loop." << std::endl;
        return std::nullopt;
    } else {
        score = icp.getFitnessScore();
        std::cout << "[SC loop] ICP fitness test passed (" << icp.getFitnessScore() << " < " << historyKeyframeFitnessScore << "). Add this SC loop." << std::endl;
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation(); // T_st
    mKF.lock(); 
    Eigen::Affine3f tWrong = PoseToAffine3f(keyframePosesUpdated[_curr_kf_idx]);
    // Eigen::Affine3f tCorrect = tWrong* correctionLidarFrame;// pre-multiplying -> successive rotation about a fixed frame
    Eigen::Affine3f tCorrect = correctionLidarFrame*tWrong;// pre-multiplying -> successive rotation about a fixed frame

    pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
    auto loopPose = keyframePosesUpdated[_loop_kf_idx];
    mKF.unlock(); 
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(loopPose.roll, loopPose.pitch, loopPose.yaw), Point3(loopPose.x,loopPose.y,loopPose.z));

    return poseFrom.between(poseTo);
} // doICPVirtualRelative

// add odom and GPS factor
void addOdomFactor(int& prev_node_idx, int& curr_node_idx)
{
    if( ! gtSAMgraphMade /* prior node */) {  // 第一帧先验
        const int init_node_idx = 0; 
        gtsam::Pose3 poseOrigin = Pose6DtoGTSAMPose3(keyframePoses.at(init_node_idx));
        // auto poseOrigin = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));

        
        {
            // prior factor 
            gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(init_node_idx, poseOrigin, priorNoise));
            initialEstimate.insert(init_node_idx, poseOrigin);
            // runISAM2opt();   
            if(evalFormat =="kitti"){            
                if(0){
                    gtsam::Pose3 pose = Pose6DtoGTSAMPose3(keyframePoses.at(curr_node_idx));       

                Point3 t = pose.translation();
                Rot3 R = pose.rotation();
                auto col1 = R.column(1); // Point3
                auto col2 = R.column(2); // Point3
                auto col3 = R.column(3); // Point3           
                groud_truth_stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << currGPS.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << currGPS.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << currGPS.z() << std::endl;

                }
                else{
                     // tmpQwl = tmpQwl.slerp(gpsPoseProportion,tmpQwl_gps);   
                Eigen::Matrix3d tmpR(currGPSQ);         

                    groud_truth_stream << tmpR(0,0) << " " << tmpR(0,1) << " " << tmpR(0,2) << " " << currGPS.x() << " "
                << tmpR(1,0) << " " << tmpR(1,1) << " " << tmpR(1,2) << " " << currGPS.y() << " "
                << tmpR(2,0) << " " << tmpR(2,1) << " " << tmpR(2,2) << " " << currGPS.z() << std::endl;
                }
                
               
                

            }       
        }   
        

        gtSAMgraphMade = true; 

        cout << "posegraph prior node " << init_node_idx << " added" << endl;
    } else /* consecutive node (and odom factor) after the prior added */ { // == keyframePoses.size() > 1 

        // 取出里程位姿
        gtsam::Pose3 poseFrom = Pose6DtoGTSAMPose3(keyframePoses.at(prev_node_idx));
        gtsam::Pose3 poseTo = Pose6DtoGTSAMPose3(keyframePoses.at(curr_node_idx));

       
        {
            // odom factor
            gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, poseFrom.between(poseTo), odomNoise));

            // gps factor 
            if(hasGPSforThisKF) {   // 当前帧有对应的GPS
                
                gtSAMgraph.add(gtsam::GPSFactor(curr_node_idx, gtsam::Point3(currGPS.x(), currGPS.y(), currGPS.z()), robustGPSNoise));
                cout << "GPS factor added at node ss" << curr_node_idx << endl;
                if(evalFormat =="kitti"){
                    if(0){
                        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(keyframePoses.at(curr_node_idx));       

                        Point3 t = pose.translation();
                        Rot3 R = pose.rotation();
                        auto col1 = R.column(1); // Point3
                        auto col2 = R.column(2); // Point3
                        auto col3 = R.column(3); // Point3           
                        groud_truth_stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << currGPS.x() << " "
                    << col1.y() << " " << col2.y() << " " << col3.y() << " " << currGPS.y() << " "
                    << col1.z() << " " << col2.z() << " " << col3.z() << " " << currGPS.z() << std::endl;
                    }
                    else{
                        Eigen::Matrix3d tmpR(currGPSQ);         

                     groud_truth_stream << tmpR(0,0) << " " << tmpR(0,1) << " " << tmpR(0,2) << " " << currGPS.x() << " "
                    << tmpR(1,0) << " " << tmpR(1,1) << " " << tmpR(1,2) << " " << currGPS.y() << " "
                    << tmpR(2,0) << " " << tmpR(2,1) << " " << tmpR(2,2) << " " << currGPS.z() << std::endl;
                    }
             
                    
                 

                }
            }

            initialEstimate.insert(curr_node_idx, poseTo);                
            // runISAM2opt();
        }
        

        // if(curr_node_idx % 100 == 0)
        //     cout << "posegraph odom node " << curr_node_idx << " added." << endl;
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

void process_pg()
{
    // Parse_GPS parseGPS;
    while(ros::ok())
    {
		while ( !odometryBuf.empty() && !fullResBuf.empty() )  // 
        {
            //
            // pop and check keyframe is or not  
            // 
			mBuf.lock();    
            // 里程计比点云旧,抛弃旧的历程   
            while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < (fullResBuf.front()->header.stamp.toSec()-0.01))
                odometryBuf.pop();
            
            // 没有里程数据了,跳出
            if (odometryBuf.empty())
            {
                mBuf.unlock();
                break;
            }

            // Time equal check
            timeLaserOdometry = odometryBuf.front()->header.stamp.toSec(); // 里程计时间
            timeLaser = fullResBuf.front()->header.stamp.toSec(); // 点云时间
            // TODO

            // laserCloudFullRes->clear(); // 没有用
            pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*fullResBuf.front(), *thisKeyFrame);
            fullResBuf.pop();

            // find nearest gps 
            double eps = 0.01; // find a gps topioc arrived within eps second 
            while (!gpsBuf.empty()) {
                auto thisGPS = gpsBuf.front();
                auto thisGPSTime = thisGPS.second;
                if( abs(thisGPSTime - timeLaserOdometry) < eps ) {
                    currGPS = thisGPS.first;
                    currGPSQ = gpsBufQ.front();
                    hasGPSforThisKF = true; 
                    break;
                } else {
                    hasGPSforThisKF = false;
                }
                gpsBuf.pop();
                gpsBufQ.pop();
            }

            Pose6D pose_curr = getOdom(odometryBuf.front()); // ros odom数据转xyzrpy
            odometryBuf.pop();
            mBuf.unlock(); 

            //
            // Early reject by counting local delta movement (for equi-spereated kf drop)
            // 
            odom_pose_prev = odom_pose_curr; // 上一帧位姿
            odom_pose_curr = pose_curr; // 当前帧位姿

        
            Pose6D dtf = diffTransformation(odom_pose_prev, odom_pose_curr);// 两帧间的相对xyzrpy

            double delta_translation = sqrt(dtf.x*dtf.x + dtf.y*dtf.y + dtf.z*dtf.z); // note: absolute value. 位移

            translationAccumulated += delta_translation; //总平移变化
            rotaionAccumulated += (dtf.roll + dtf.pitch + dtf.yaw); // sum just naive approach. 总角度变化 

            // 判断平移或者角度变化是否大于阈值
            if( translationAccumulated > keyframeMeterGap || rotaionAccumulated > keyframeRadGap ) {
                isNowKeyFrame = true;
                translationAccumulated = 0.0; // reset 
                rotaionAccumulated = 0.0; // reset 
            } else {
                isNowKeyFrame = false;
            }

            // 非关键帧
            if( ! isNowKeyFrame ) 
                continue; 



            //
            // Save data and Add consecutive node 
            //
            pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
#if dowmSample
            
            downSizeFilterScancontext.setInputCloud(thisKeyFrame);
            downSizeFilterScancontext.filter(*thisKeyFrameDS);
#else
            thisKeyFrameDS = thisKeyFrame;

#endif
            mKF.lock(); 
            keyframeLaserClouds.push_back(thisKeyFrameDS); // 关键帧点云
            keyframePoses.push_back(odom_pose_curr); // 当前位姿
            keyframePosesUpdated.push_back(odom_pose_curr); // init
            PointType tmpPoint;
            tmpPoint.x = odom_pose_curr.x;
            tmpPoint.y = odom_pose_curr.y;
            tmpPoint.z = odom_pose_curr.z;
            tmpPoint.intensity = timeLaser;
            cloudKeyPoses3D->push_back(tmpPoint);
            keyframeTimes.push_back(timeLaserOdometry); // 时间戳
            
            // sc 添加关键帧
            // pcl::PointCloud<pcl::PointXYZI>::Ptr tmpThisKeyFrameDS(new pcl::PointCloud<pcl::PointXYZI>());
            // int tmp_count = thisKeyFrameDS->points.size();
            // tmpThisKeyFrameDS->resize(tmp_count);
            // for(int i=0; i<tmp_count; ++i ){
            //     tmpThisKeyFrameDS->points[i].x = thisKeyFrameDS->points[i].x;
            //     tmpThisKeyFrameDS->points[i].y = thisKeyFrameDS->points[i].y;
            //     tmpThisKeyFrameDS->points[i].z = thisKeyFrameDS->points[i].z;
            //     tmpThisKeyFrameDS->points[i].intensity= thisKeyFrameDS->points[i].intensity;
            // }

            scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameDS);

         
            mKF.unlock(); 
            int prev_node_idx = keyframePoses.size() - 2; 
            int curr_node_idx = keyframePoses.size() - 1; // becuase cpp starts with 0 (actually this index could be any number, but for simple implementation, we follow sequential indexing)
    
            addOdomFactor(prev_node_idx, curr_node_idx);
            addLoopFactor();
            runISAM2opt();
            pubPath();
            // if want to print the current graph, use gtSAMgraph.print("\nFactor Graph:\n");

            // save utility  保存当前关键帧
            std::string curr_node_idx_str = padZeros(curr_node_idx);
            pcl::io::savePCDFileBinary(pgScansDirectory + curr_node_idx_str + ".pcd", *thisKeyFrame); // scan 
            pgTimeSaveStream << timeLaser << std::endl; // path 
        }
        // if(keyframePoses.size()>10){
        //     runISAM2opt();
        // }
        
      
        // scan context detector is running in another thread (in constant Hz, e.g., 1 Hz)
        // pub path and point cloud in another thread

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_pg

void performSCLoopClosure(void)
{
    if( int(keyframePoses.size()) < scManager.NUM_EXCLUDE_RECENT) // do not try too early 
        return;

    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
    int SCclosestHistoryFrameID = detectResult.first;  // 找到对应的历史关键帧
    if( SCclosestHistoryFrameID != -1 ) { 
        const int prev_node_idx = SCclosestHistoryFrameID;
        mKF.lock(); 
        const int curr_node_idx = keyframePoses.size() - 1; // because cpp starts 0 and ends n-1
        mKF.unlock(); 
          // 检测这个关键帧是否已经在闭环容器里了，防止重复闭环
        auto it = loopIndexContainer.find(curr_node_idx);
        if (it != loopIndexContainer.end())
         return;
        cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;

        float score = 1000;
        auto relative_pose_optional = doICPVirtualRelative(prev_node_idx, curr_node_idx,score); // ICP验证闭环
        if(score<0.3) { // 添加闭环因子
            gtsam::Vector Vector6(6);
            Vector6 << score, score, score, score, score, score;
            noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);
            loopIndexQueue.push_back(make_pair(curr_node_idx, prev_node_idx));
            loopPoseQueue.push_back(relative_pose_optional.value());
            loopNoiseQueue.push_back(constraintNoise);

            mBuf.lock();        
            loopIndexContainer[curr_node_idx] = prev_node_idx;
            // addding actual 6D constraints in the other thread, icp_calculation.
            mBuf.unlock();

            
        } 

       
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
    
    // 在历史关键帧中找，时间相差超过阈值即可 && 应该还得关键帧序号相差超过一定阈值才行

    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
        int id = pointSearchIndLoop[i];
        if (abs(copy_cloudKeyPoses3D->points[id].intensity - timeLaser) > 30.0 && loopKeyCur-id>30)
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

void performRSLoopClosure(void)
{
    if( int(keyframePoses.size()) < scManager.NUM_EXCLUDE_RECENT) // do not try too early 
        return;


    int prev_node_idx ;  // 闭环帧id
    int curr_node_idx ; // 当前帧id
    mKF.lock(); 
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    mKF.unlock(); 

    bool isValidRSloopFactor = detectLoopClosureDistance(&curr_node_idx, &prev_node_idx); // 开启rs

    if(isValidRSloopFactor ) { 

        cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;

        float score =1000.0;
        auto relative_pose_optional = doICPVirtualRelative(prev_node_idx, curr_node_idx,score); // ICP验证闭环
        if(score<0.3) { // 添加闭环因子
            gtsam::Vector Vector6(6);
            Vector6 << score, score, score, score, score, score;
            noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);
            loopIndexQueue.push_back(make_pair(curr_node_idx, prev_node_idx));
            loopPoseQueue.push_back(relative_pose_optional.value());
            loopNoiseQueue.push_back(constraintNoise);

            mBuf.lock();        
            loopIndexContainer[curr_node_idx] = prev_node_idx;
            // addding actual 6D constraints in the other thread, icp_calculation.
            mBuf.unlock();
        } 

    }
}


// 可视化闭环关系，将闭环的两个帧连线

void visualizeLoopClosure()
{
    visualization_msgs::MarkerArray markerArray;
    // loop nodes
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = "world";
    markerNode.header.stamp = ros::Time().fromSec(timeLaser);
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
    markerEdge.header.stamp = ros::Time().fromSec(timeLaser);
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
        p.x = keyframePosesUpdated[key_cur].x;
        p.y = keyframePosesUpdated[key_cur].y;
        p.z = keyframePosesUpdated[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = keyframePosesUpdated[key_pre].x;
        p.y = keyframePosesUpdated[key_pre].y;
        p.z = keyframePosesUpdated[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge.publish(markerArray);
}
void process_lcd(void)
{
    float loopClosureFrequency = 1.0; // can change 
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        rate.sleep();
        performSCLoopClosure();
        // performRSLoopClosure(); // TODO
        visualizeLoopClosure();
    }
} // process_lcd







void pubMap(void)
{
    int SKIP_FRAMES = 2; // sparse map visulalization to save computations 
    int counter = 0;

    laserCloudMapPGO->clear();

    mKF.lock(); 
    // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()); node_idx++) {
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) {
        if(counter % SKIP_FRAMES == 0) {
            *laserCloudMapPGO += *local2global(keyframeLaserClouds[node_idx], keyframePosesUpdated[node_idx]);
        }
        counter++;
    }
    mKF.unlock(); 
#if dowmSample
    downSizeFilterMapPGO.setInputCloud(laserCloudMapPGO);
    downSizeFilterMapPGO.filter(*laserCloudMapPGO);
#endif


    sensor_msgs::PointCloud2 laserCloudMapPGOMsg;
    pcl::toROSMsg(*laserCloudMapPGO, laserCloudMapPGOMsg);
    laserCloudMapPGOMsg.header.frame_id = "world";
    pubMapAftPGO.publish(laserCloudMapPGOMsg);
}

void process_viz_map(void)
{
    float vizmapFrequency = 0.2; // 0.1 means run onces every 10s
    ros::Rate rate(vizmapFrequency);
    while (ros::ok()) {
        rate.sleep();
        if(recentIdxUpdated > 1) {
            pubMap();
            
        }
    }
     // 保存轨迹
    if(evalFormat =="kitti"){
        saveOptimizedVerticesKittiformat(isamCurrentEstimate, pgKITTIformat); // pose
        saveOdometryVerticesKittiformat(odomKITTIformat);// pose
    }
    else if(evalFormat =="tum"){
        saveOptimizedVerticesTUMformat(isamCurrentEstimate, pgKITTIformat); // pose
        saveOdometryVerticesTUMformat(odomKITTIformat); // pose
    }
    
    //save globalMap
    pcl::io::savePCDFileBinary(save_directory  + "globalMap.pcd", *laserCloudMapPGO); // scan 

} // pointcloud_viz
};

int main(int argc, char **argv)
{
  
	ros::init(argc, argv, "laserPGO");

    Map_Build MB;

	std::thread posegraph_slam {&Map_Build::process_pg,&MB}; // pose graph construction  添加里程因子和GPS因子
	std::thread lc_detection {&Map_Build::process_lcd,&MB}; // loop closure detection  

	// std::thread isam_update {&Map_Build::process_isam,&MB}; // if you want to call less isam2 run (for saving redundant computations and no real-time visulization is required), uncommment this and comment all the above runisam2opt when node is added. 

	std::thread viz_map {&Map_Build::process_viz_map,&MB}; // visualization - map (low frequency because it is heavy)
	// std::thread viz_path {&Map_Build::process_viz_path,&MB}; // visualization - path (high frequency)


 	ros::spin();
    posegraph_slam.join();
    lc_detection.join();
    // isam_update.join();
    viz_map.join();
    // viz_path.join();
	return 0;
}
