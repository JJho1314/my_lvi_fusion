#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver/CustomMsg.h>

#include "lvi_fusion/cloud_info.h"
#include <opencv2/opencv.hpp>
#include <pcl/filters/voxel_grid.h>
using namespace std;

// 去掉无穷点
#define IS_VALID(a) ((abs(a) > 1e8) ? true : false)

namespace velodyne_ros
{
    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D;
        float intensity;
        float time;
        std::uint16_t ring;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
} // namespace velodyne_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
                                      (float, time, time)(std::uint16_t, ring, ring))


namespace rslidar_ros {
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace velodyne_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(rslidar_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
                                    )

typedef pcl::PointXYZINormal PointType;

ros::Publisher pub_full, pub_surf, pub_corn;

enum LID_TYPE
{
    MID, 
    HORIZON,
    VELO16,
    OUST64,
    rslidar
};

enum Feature
{
    Nor,
    Poss_Plane,
    Real_Plane,
    Edge_Jump,
    Edge_Plane,
    Wire,
    ZeroPoint
};
enum Surround
{
    Prev,
    Next
};
enum E_jump
{
    Nr_nor,
    Nr_zero,
    Nr_180,
    Nr_inf,
    Nr_blind
};

// feature type
struct orgtype
{
    double  range;  // 点云点到原点的距离
    double  dista; // 当前点与同一scan的下一个点的距离
    double  angle[ 2 ];
    double  intersect;
    E_jump  edj[ 2 ];
    Feature ftype;
    orgtype()
    {
        range = 0;
        edj[ Prev ] = Nr_nor;
        edj[ Next ] = Nr_nor;
        ftype = Nor;
        intersect = 2;
    }
};

const double rad2deg = 180 * M_1_PI;

int    lidar_type;
double blind, inf_bound;
int    N_SCANS;
int Horizon_SCAN;// 列线数groundScanInd
int    group_size;  // 群的最小大小
double disA, disB;  // 用于计算群尺寸的系数
double limit_maxmid, limit_midmin, limit_maxmin;
double p2l_ratio;
double jump_up_limit, jump_down_limit;
double cos160;
double edgea, edgeb;
double smallp_intersect, smallp_ratio;
int    point_filter_num;
int    g_if_using_raw_point = 1;
int    g_LiDAR_sampling_point_step = 3;
float  time_scale_ = 1e-3;
bool   given_offset_time_ =false;

//曲率值和序号
struct smoothness_t{ 
    float value;
    size_t ind;
};

//曲率值对比仿函数
struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

PointType nanPoint;
pcl::PointCloud<PointType>::Ptr fullCloud;
lvi_fusion::cloud_info cloudInfo;
cv::Mat rangeMat; 
pcl::PointCloud<PointType>::Ptr extractedCloud; // 初步筛选后用于提取的点云
int *cloudNeighborPicked;// 是否已经被挑选出来，即去掉已经筛选过的点的标志 1为已经筛选过 0为还未被筛选
int *cloudLabel;  // 点云类型标记 初始化为0，平面点为-1，角点为1 ，// TODO 其实好像也没什么用啊反正都已经放到对应的队列进去了，这个根本没用嘛
float *cloudCurvature;
std::vector<smoothness_t> cloudSmoothness;
int area_num =6;
float edgeThreshold =1.;
float surfThreshold = 0.1;
float ang_res_x;   // 角度分辨率
float ang_res_y;
pcl::VoxelGrid<PointType> Voxel_Sample;
float Vertical_angle= 41.33;


void   mid_handler( const sensor_msgs::PointCloud2::ConstPtr &msg );
void   horizon_handler( const livox_ros_driver::CustomMsg::ConstPtr &msg );
void   velo16_handler( const sensor_msgs::PointCloud2::ConstPtr &msg );
void   velo16_handler1( const sensor_msgs::PointCloud2::ConstPtr &msg );
void   rs32_handler( const sensor_msgs::PointCloud2::ConstPtr &msg );
void   oust64_handler( const sensor_msgs::PointCloud2::ConstPtr &msg );
void   give_feature( pcl::PointCloud< PointType > &pl, vector< orgtype > &types, pcl::PointCloud< PointType > &pl_corn,
                     pcl::PointCloud< PointType > &pl_surf );


void   pub_func( pcl::PointCloud< PointType > &pl, ros::Publisher pub, const ros::Time &ct );
int    plane_judge( const pcl::PointCloud< PointType > &pl, vector< orgtype > &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct );
bool   small_plane( const pcl::PointCloud< PointType > &pl, vector< orgtype > &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct );
bool   edge_jump_judge( const pcl::PointCloud< PointType > &pl, vector< orgtype > &types, uint i, Surround nor_dir );

// loam param and callback
int main( int argc, char **argv )
{
    ros::init( argc, argv, "feature_extract" );
    ros::NodeHandle n;

    n.param< int >( "Lidar_front_end/lidar_type", lidar_type, 0 );
    n.param< double >( "Lidar_front_end/blind", blind, 2.0 );
    n.param< double >( "Lidar_front_end/inf_bound", inf_bound, 4 );
    n.param< int >( "Lidar_front_end/N_SCANS", N_SCANS, 1 );
    n.param< int >( "Lidar_front_end/Horizon_SCAN", Horizon_SCAN, 1800 );
    n.param< int >( "Lidar_front_end/group_size", group_size, 8 );
    n.param< double >( "Lidar_front_end/disA", disA, 0.01 );
    n.param< double >( "Lidar_front_end/disB", disB, 0.1 );
    n.param< double >( "Lidar_front_end/p2l_ratio", p2l_ratio, 225 );
    n.param< double >( "Lidar_front_end/limit_maxmid", limit_maxmid, 6.25 );
    n.param< double >( "Lidar_front_end/limit_midmin", limit_midmin, 6.25 );
    n.param< double >( "Lidar_front_end/limit_maxmin", limit_maxmin, 3.24 );
    n.param< double >( "Lidar_front_end/jump_up_limit", jump_up_limit, 170.0 );
    n.param< double >( "Lidar_front_end/jump_down_limit", jump_down_limit, 8.0 );
    n.param< double >( "Lidar_front_end/cos160", cos160, 160.0 );
    n.param< double >( "Lidar_front_end/edgea", edgea, 2 );
    n.param< double >( "Lidar_front_end/edgeb", edgeb, 0.1 );
    n.param< double >( "Lidar_front_end/smallp_intersect", smallp_intersect, 172.5 );
    n.param< double >( "Lidar_front_end/smallp_ratio", smallp_ratio, 1.2 );
    n.param< int >( "Lidar_front_end/point_filter_num", point_filter_num, 1 );  // velodyne 4
    n.param< int >( "Lidar_front_end/point_step", g_LiDAR_sampling_point_step, 3 );
    n.param< int >( "Lidar_front_end/using_raw_point", g_if_using_raw_point, 1 );

    jump_up_limit = cos( jump_up_limit / 180 * M_PI );
    jump_down_limit = cos( jump_down_limit / 180 * M_PI );
    cos160 = cos( cos160 / 180 * M_PI );
    smallp_intersect = cos( smallp_intersect / 180 * M_PI );

    // 旋转激光的一些初始化
    nanPoint.x = std::numeric_limits<float>::quiet_NaN();// 返回目标类型的安静NAN的表示
    nanPoint.y = std::numeric_limits<float>::quiet_NaN();
    nanPoint.z = std::numeric_limits<float>::quiet_NaN();
    nanPoint.intensity = -1;

    fullCloud.reset(new pcl::PointCloud<PointType>());
    extractedCloud.reset(new pcl::PointCloud<PointType>());
    fullCloud->points.resize(N_SCANS*Horizon_SCAN);

    cloudInfo.startRingIndex.assign(N_SCANS, 0);
    cloudInfo.endRingIndex.assign(N_SCANS, 0);
    cloudInfo.pointColInd.assign(N_SCANS*Horizon_SCAN, 0);
    cloudInfo.pointRange.assign(N_SCANS*Horizon_SCAN, 0);

    cloudSmoothness.resize(N_SCANS*Horizon_SCAN);
    cloudCurvature = new float[N_SCANS*Horizon_SCAN];
    cloudNeighborPicked = new int[N_SCANS*Horizon_SCAN];
    cloudLabel = new int[N_SCANS*Horizon_SCAN];

    Voxel_Sample.setLeafSize(0.4, 0.4, 0.4);

    ang_res_x = 360.0/float(Horizon_SCAN);
    ang_res_y = Vertical_angle/float(N_SCANS-1);

    ros::Subscriber sub_points;
    cout << "lidar_type" <<endl;
    switch ( lidar_type )
    {
    case MID:
        printf( "MID40\n" );
        sub_points = n.subscribe( "/livox/lidar", 1000, mid_handler, ros::TransportHints().tcpNoDelay() );
        break;

    case HORIZON:
        printf( "HORIZON\n" );
        sub_points = n.subscribe( "/livox/lidar", 1000, horizon_handler, ros::TransportHints().tcpNoDelay() );
        break;

    case VELO16:
        printf( "VELO16\n" );
#if 1     
        sub_points = n.subscribe( "/velodyne_points", 1000, velo16_handler1, ros::TransportHints().tcpNoDelay() );
#else
        sub_points = n.subscribe( "/velodyne_points", 1000, velo16_handler, ros::TransportHints().tcpNoDelay() );
#endif
        break;

    case OUST64:
        printf( "OUST64\n" ); 
        sub_points = n.subscribe( "/os1_cloud_node1/points", 1000, oust64_handler, ros::TransportHints().tcpNoDelay() );
        break;

    case rslidar:
        printf( "RSLIDAR\n" );
  
        sub_points = n.subscribe( "/rslidar_points", 1000, rs32_handler, ros::TransportHints().tcpNoDelay() );

        break;

    default:
        printf( "Lidar type is wrong.\n" );
        exit( 0 );
        break;
    }

    pub_full = n.advertise< sensor_msgs::PointCloud2 >( "/laser_cloud", 100 );
    pub_surf = n.advertise< sensor_msgs::PointCloud2 >( "/laser_cloud_flat", 100 );
    pub_corn = n.advertise< sensor_msgs::PointCloud2 >( "/laser_cloud_sharp", 100 );

    ros::spin();
    return 0;
}

double vx, vy, vz;
void   mid_handler( const sensor_msgs::PointCloud2::ConstPtr &msg )
{
    pcl::PointCloud< PointType > pl;
    pcl::fromROSMsg( *msg, pl );

    pcl::PointCloud< PointType > pl_corn, pl_surf;
    vector< orgtype >            types;
    uint                         plsize = pl.size() - 1;
    pl_corn.reserve( plsize );
    pl_surf.reserve( plsize );
    types.resize( plsize + 1 );

    for ( uint i = 0; i < plsize; i++ )
    {
        types[ i ].range = pl[ i ].x;
        vx = pl[ i ].x - pl[ i + 1 ].x;
        vy = pl[ i ].y - pl[ i + 1 ].y;
        vz = pl[ i ].z - pl[ i + 1 ].z;
        types[ i ].dista = vx * vx + vy * vy + vz * vz;
    }
    // plsize++;
    types[ plsize ].range = sqrt( pl[ plsize ].x * pl[ plsize ].x + pl[ plsize ].y * pl[ plsize ].y );

    give_feature( pl, types, pl_corn, pl_surf );

    ros::Time ct( ros::Time::now() );
    pub_func( pl, pub_full, msg->header.stamp );
    pub_func( pl_surf, pub_surf, msg->header.stamp );
    pub_func( pl_corn, pub_corn, msg->header.stamp );
}

void horizon_handler( const livox_ros_driver::CustomMsg::ConstPtr &msg )
{
   
    double                                 t1 = omp_get_wtime();
    vector< pcl::PointCloud< PointType > > pl_buff( N_SCANS );// 按线分的点云，去掉了重复的点
    vector< vector< orgtype > >            typess( N_SCANS );
    pcl::PointCloud< PointType >           pl_full, // 全部的原始点云，use curvature as time of each laser points
                                           pl_corn, 
                                           pl_surf; // 降采样的原始点云

    uint plsize = msg->point_num;  // all point's number

    pl_corn.reserve( plsize );
    pl_surf.reserve( plsize );
    pl_full.resize( plsize );

    for ( int i = 0; i < N_SCANS; i++ )
    {
        pl_buff[ i ].reserve( plsize );
    }
    // ANCHOR - remove nearing pts.
    for ( uint i = 1; i < plsize; i++ )
    {
        // clang-format off
        if ( ( msg->points[ i ].line < N_SCANS ) 
            && ( !IS_VALID( msg->points[ i ].x ) )  // 无效点
            && ( !IS_VALID( msg->points[ i ].y ) ) 
            && ( !IS_VALID( msg->points[ i ].z ) )
            && msg->points[ i ].x > 0.7 ) // 盲区点
        {
            // https://github.com/Livox-SDK/Livox-SDK/wiki/Livox-SDK-Communication-Protocol
            // See [3.4 Tag Information] 
            if ( ( msg->points[ i ].x > 2.0 )
                && ( ( ( msg->points[ i ].tag & 0x03 ) != 0x00 )  ||  ( ( msg->points[ i ].tag & 0x0C ) != 0x00 ) )
                )
            {
                // Remove the bad quality points
                continue;
            }
            // clang-format on
            pl_full[i].x = msg->points[i].x;
            pl_full[i].y = msg->points[i].y;
            pl_full[i].z = msg->points[i].z;
            pl_full[i].intensity = msg->points[i].reflectivity;

            pl_full[i].curvature = msg->points[i].offset_time / float(1000000); // use curvature as time of each laser points

            if ((std::abs(pl_full[i].x - pl_full[i - 1].x) > 1e-7) || (std::abs(pl_full[i].y - pl_full[i - 1].y) > 1e-7) ||
                (std::abs(pl_full[i].z - pl_full[i - 1].z) > 1e-7))
            {
                pl_buff[msg->points[i].line].push_back(pl_full[i]);
            }
        }
    }
    if (pl_buff.size() != N_SCANS)
    {
        return;
    }
    if (pl_buff[0].size() <= 7)
    {
        return;
    }

    // 遍历每条scan
    for (int j = 0; j < N_SCANS; j++)
    {
        pcl::PointCloud<PointType> &pl = pl_buff[j];
        vector<orgtype> &types = typess[j];
        plsize = pl.size();
        if (plsize < 7)
        {
            continue;
        }
        types.resize(plsize);
        plsize--;
        for (uint pt_idx = 0; pt_idx < plsize; pt_idx++)
        {
            types[pt_idx].range = pl[pt_idx].x * pl[pt_idx].x + pl[pt_idx].y * pl[pt_idx].y;
            vx = pl[pt_idx].x - pl[pt_idx + 1].x;
            vy = pl[pt_idx].y - pl[pt_idx + 1].y;
            vz = pl[pt_idx].z - pl[pt_idx + 1].z;
            // std::cout<<vx<<" "<<vx<<" "<<vz<<" "<<std::endl;
        }
        // plsize++;
        types[plsize].range = pl[plsize].x * pl[plsize].x + pl[plsize].y * pl[plsize].y;
        give_feature(pl, types, pl_corn, pl_surf);
    }
    if (pl_surf.points.size() < 100)
    {
        return;
    }
    ros::Time ct;
    ct.fromNSec(msg->timebase);
    pub_func(pl_full, pub_full, msg->header.stamp);
    pub_func(pl_surf, pub_surf, msg->header.stamp);
    pub_func(pl_corn, pub_corn, msg->header.stamp);
}

int orders[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};

// 当前点到原点的距离
template <typename point_type>
inline float pointDistance(point_type p)
{
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

template <typename T>
inline bool has_nan(T point)
{

    // remove nan point, or the feature assocaion will crash, the surf point will containing nan points
    // pcl remove nan not work normally
    // ROS_ERROR("Containing nan point!");
    if (isnan(point.x) || isnan(point.y) || isnan(point.z))
    {
        return true;
    }
    else
    {
        return false;
    }
}

//把有range值的fullcloud 赋值给 extractedCloud
//记录startRingIndex、endRingIndex、pointColInd、pointRange
void cloudExtraction()
{

    int count = 0;
    // extract segmented cloud for lidar odometry

    for (int i = 0; i < N_SCANS; ++i) // 16
    {
        cloudInfo.startRingIndex[i] = count - 1 + 5;

        for (int j = 0; j < Horizon_SCAN; ++j) // 1800
        {

            if (rangeMat.at<float>(i, j) != FLT_MAX) //   labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1 rangeMat.at<float>(i,j)!=FLT_MAX
            {

                // if (labelMat.at<int>(i,j) == 999999){
                //     continue;
                // }
                // 如果是地面点,对于列数不为5的倍数的，直接跳过不处理
                // 相当于地面点降采样
                // if (groundMat.at<int8_t>(i,j) == 1){
                //     if (j%5!=0 && j>5 && j<Horizon_SCAN-5)
                //         continue;
                // }
                // mark the points' column index for marking occlusion later

                // cloudInfo.segmentedCloudGroundFlag[count] = (groundMat.at<int8_t>(i,j) == 1);

                cloudInfo.pointColInd[count] = j;
                // cloudInfo.pointRowInd[count] = i;
                // save range info

                cloudInfo.pointRange[count] = rangeMat.at<float>(i, j);
                // save extracted cloud

                extractedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);

                // size of extracted cloud
                ++count;
            }
        }
        cloudInfo.endRingIndex[i] = count - 1 - 5;
    }
}
//计算所有点的曲率
void calculateSmoothness()
{
    int cloudSize = extractedCloud->points.size();

    for (int i = 5; i < cloudSize - 5; ++i)
    {
        // 前后各5个点与当前点的range差
        float diffRange = cloudInfo.pointRange[i - 5] + cloudInfo.pointRange[i - 4] + cloudInfo.pointRange[i - 3] + cloudInfo.pointRange[i - 2] + cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i] * 10 + cloudInfo.pointRange[i + 1] + cloudInfo.pointRange[i + 2] + cloudInfo.pointRange[i + 3] + cloudInfo.pointRange[i + 4] + cloudInfo.pointRange[i + 5];

        // 曲率
        cloudCurvature[i] = diffRange * diffRange; // diffX * diffX + diffY * diffY + diffZ * diffZ;

        cloudNeighborPicked[i] = 0; // 初始化赋值
        cloudLabel[i] = 0;          // 初始化赋值
        // cloudSmoothness for sorting
        cloudSmoothness[i].value = cloudCurvature[i];
        cloudSmoothness[i].ind = i;
    }
}

// 剔除被遮挡的点不参与特征提取
//相邻两点range差距大则设为不进行特征提取的状态，差距过大，周围点也设置为不进行特征提取的状态
// 标记为1代表这些点已经被筛选，不再参与特征点的提取
void markOccludedPoints()
{
    int cloudSize = extractedCloud->points.size();
    // mark occluded points and parallel beam points
    for (int i = 5; i < cloudSize - 6; ++i)
    {

        // occluded points
        float depth1 = cloudInfo.pointRange[i];
        float depth2 = cloudInfo.pointRange[i + 1];
        //两个点的列差值
        int columnDiff = std::abs(int(cloudInfo.pointColInd[i + 1] - cloudInfo.pointColInd[i]));
        // 两个点的水平距离分明很近，但是深度却差的很大，说明这两个点之间存在遮挡
        if (columnDiff < 10)
        { // 10 pixel diff in range image
            // 哪边深度深，哪边被标记为已经筛选过，即不进行特征提取
            if (depth1 - depth2 > 0.3)
            { // TODO if 0.3 is too small
                cloudNeighborPicked[i - 5] = 1;
                cloudNeighborPicked[i - 4] = 1;
                cloudNeighborPicked[i - 3] = 1;
                cloudNeighborPicked[i - 2] = 1;
                cloudNeighborPicked[i - 1] = 1;
                cloudNeighborPicked[i] = 1;
            }
            else if (depth2 - depth1 > 0.3)
            {
                cloudNeighborPicked[i + 1] = 1;
                cloudNeighborPicked[i + 2] = 1;
                cloudNeighborPicked[i + 3] = 1;
                cloudNeighborPicked[i + 4] = 1;
                cloudNeighborPicked[i + 5] = 1;
                cloudNeighborPicked[i + 6] = 1;
            }
        }
        // parallel beam
        // 平行光束
        float diff1 = std::abs(float(cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i]));
        float diff2 = std::abs(float(cloudInfo.pointRange[i + 1] - cloudInfo.pointRange[i]));
        // 两个相邻光束的range差的很大就可以认为是平行光速
        if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
            cloudNeighborPicked[i] = 1;
    }
}

// void extractFeatures()进行特征抽取，然后分别保存到cornerPointsSharp等等队列中去。
// 函数首先清空了cornerCloud,surfaceCloud
// 然后对cloudSmoothness队列中sp到ep之间的点的曲率数据进行从小到大的排列。
// 曲率最大的20个点作为角点cornerCloud，满足平面阈值的点都作为平面点surfaceCloudScan
// 最后，因为平面点云太多时，计算量过大，因此需要对点云进行下采样surfaceCloudScanDS，减少计算量。
// 下采样之后的点作为surfaceCloud
void extractFeatures(pcl::PointCloud<PointType>::Ptr cornerCloud, pcl::PointCloud<PointType>::Ptr surfaceCloud)
{

    // 一个水平线束的平面点 和其降采样
    pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

    for (int i = 0; i < N_SCANS; i++)
    {
        surfaceCloudScan->clear();

        for (int j = 0; j < area_num; j++)
        {
            // 和loam中一样把点云分成了6段分别处理，确保特征点提取均匀
            // 这里就是差值分成了6块区域
            int sp = (cloudInfo.startRingIndex[i] * (area_num - j) + cloudInfo.endRingIndex[i] * j) / area_num;
            int ep = (cloudInfo.startRingIndex[i] * (area_num - 1 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / area_num - 1;

            if (sp >= ep)
                continue;

            // 按照曲率从小到大排序
            std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, by_value());

            int largestPickedNum = 0; // 被提取特征点的数量
            for (int k = ep; k >= sp; k--)
            {
                // 因为上面对cloudSmoothness进行了一次从小到大排序，所以ind不一定等于k了
                int ind = cloudSmoothness[k].ind;
                if (cloudNeighborPicked[ind] == 0          // 还未被筛选
                    && cloudCurvature[ind] > edgeThreshold //大于边的阈值
                    // && cloudInfo.segmentedCloudGroundFlag[ind] == false  // 不在地面点中找
                )
                {
                    if (k % 2 != 0)
                        continue;
                    largestPickedNum++;
                    if (largestPickedNum <= 20)
                    {                        // 曲率最大的20个点作为角点  //注意这里的k是倒着来的
                        cloudLabel[ind] = 1; //角点标记为1

                        cornerCloud->push_back(extractedCloud->points[ind]);
                        cornerCloud->points.back().normal_x = 1;
                    }
                    else
                    {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1; //标记为已经筛选过
                    // 把领域的点也标记为已经筛选过的点，应该是为了防止太密集吧
                    for (int l = 1; l <= 5; l++)
                    {
                        int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                        if (columnDiff > 10)
                            break;
                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                        if (columnDiff > 10)
                            break;
                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            //提取所有满足阈值的平面点
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSmoothness[k].ind;
                if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold
                    // && cloudInfo.segmentedCloudGroundFlag[ind] == true // 只在地面点中找
                )
                {

                    cloudLabel[ind] = -1;
                    cloudNeighborPicked[ind] = 1;

                    for (int l = 1; l <= 5; l++)
                    {

                        int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                        if (columnDiff > 10)
                            break;

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {

                        int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                        if (columnDiff > 10)
                            break;

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }
            int largestPickedSurfNum = 0; // 被提取特征点的数量
            for (int k = sp; k <= ep; k++)
            {
                if (k % 2 != 0)
                    continue;
                largestPickedSurfNum++;
                if (largestPickedSurfNum <= 200)
                { // 曲率最大的20个点作为角点  //注意这里的k是倒着来的
                    if (cloudLabel[k] <= 0)
                    {
                        // extractedCloud->points[k].normal_x = 2;
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                        surfaceCloudScan->points.back().normal_x = 2;
                    }
                }
                else
                {
                    break;
                }
            }
        }

        surfaceCloudScanDS->clear();

        Voxel_Sample.setInputCloud(surfaceCloudScan);
        Voxel_Sample.filter(*surfaceCloudScanDS);

        *surfaceCloud += *surfaceCloudScanDS;

        // !PCL库的bug，降采样后normal_x被置1了
        for (int i = 0; i < surfaceCloud->points.size(); i++)
        {
            surfaceCloud->points[i].normal_x = 2;
        }
    }
}
pcl::PointCloud<PointType>::Ptr feats_surf(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr feats_corn(new pcl::PointCloud<PointType>());
void velo16_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    rangeMat = cv::Mat(N_SCANS, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
    std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
    extractedCloud->clear();
    cloudInfo.startRingIndex.clear();
    cloudInfo.endRingIndex.clear();
    cloudInfo.pointColInd.clear();
    cloudInfo.pointRange.clear();

    pcl::PointCloud<PointType>::Ptr pl_corn, pl_surf;
    pl_corn.reset(new pcl::PointCloud<PointType>());
    pl_surf.reset(new pcl::PointCloud<PointType>());

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);

    int plsize = pl_orig.points.size();
    pl_corn->reserve(plsize);
    pl_surf->reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 3.61; // scan angular velocity
    std::vector<bool> is_first(N_SCANS, true);
    std::vector<double> yaw_fp(N_SCANS, 0.0);   // yaw of first scan point
    std::vector<float> yaw_last(N_SCANS, 0.0);  // yaw of last scan point
    std::vector<float> time_last(N_SCANS, 0.0); // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0)
    { // 自身带时间
        given_offset_time_ = true;
    }
    else
    { // 自身不带时间
        given_offset_time_ = false;
        double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578; // 第一个点的yaw
        double yaw_end = yaw_first;                                                    // 最后一个点的yaw
        int layer_first = pl_orig.points[0].ring;                                      // 第一个点所在的层
        for (uint i = plsize - 1; i > 0; i--)
        { // 倒排得到和第一个所在同一层的最后一个点
            if (pl_orig.points[i].ring == layer_first)
            {
                yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
                break;
            }
        }
    }

    for (int i = 0; i < plsize; i++)
    {
        PointType added_pt;

        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_scale_; // curvature unit: ms

        // 计算每个点的时间
        if (!given_offset_time_)
        {
            int layer = pl_orig.points[i].ring;
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

            if (is_first[layer])
            {
                yaw_fp[layer] = yaw_angle;
                is_first[layer] = false;
                added_pt.curvature = 0.0;
                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.curvature;
                continue;
            }

            // compute offset time
            if (yaw_angle <= yaw_fp[layer])
            {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
            }
            else
            {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
            }

            if (added_pt.curvature < time_last[layer])
                added_pt.curvature += 360.0 / omega_l;

            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.curvature;
        }

        if (i % point_filter_num == 0)
        {
            // 获得到原点的距离
            float range = pointDistance(added_pt);
            if (range < blind)
                continue;
            if (has_nan(added_pt))
                continue;
            int rowIdn;
            rowIdn = pl_orig.points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCANS)
                continue;
            // 如果有降采样，则每隔downsampleRate个取一个
            // if (rowIdn % downsampleRate != 0)
            //     continue;

            if (fabs(added_pt.x) < 1.1 && fabs(added_pt.y) < 2.5)
                continue;

            // 该点的水平角度
            float horizonAngle = atan2(added_pt.x, added_pt.y) * 180 / M_PI;

            // 水平分辨率

            // 算在哪个竖直线id上
            int columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;
            // 已经赋过值则跳过
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;
            rangeMat.at<float>(rowIdn, columnIdn) = range;
            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = added_pt;
        }
    }

    cloudExtraction();

    calculateSmoothness();

    markOccludedPoints();

    extractFeatures(pl_corn, pl_surf);

    *pl_surf += *pl_corn;
    // pl_surf->points.insert(pl_surf->points.end(),pl_corn->points.begin(),pl_corn->points.end());

    // feats_surf->clear();
    // feats_corn->clear();
    // for(int i=0;i<pl_surf->points.size();i++){

    //     if(pl_surf->points[i].normal_x==2){
    //         feats_corn->push_back(pl_surf->points[i]);
    //     }
    //     else{

    //         feats_surf->push_back(pl_surf->points[i]);
    //     }
    // }
    // cout << "pl_surf->points.size()"<< pl_surf->points.size()<<endl;
    // cout << "feats_corn->points.size()"<< feats_corn->points.size()<<endl;
    // cout << "feats_surf->points.size()"<< feats_surf->points.size()<<endl;

    pub_func(*fullCloud, pub_full, msg->header.stamp);

    pub_func(*pl_surf, pub_surf, msg->header.stamp);

    pub_func(*pl_corn, pub_corn, msg->header.stamp);
}

void rs32_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    double t1 = omp_get_wtime();

    pcl::PointCloud<PointType> pl_full, // 全部的原始点云，use curvature as time of each laser points
        pl_corn,
        pl_surf; // 降采样的原始点云

    pcl::PointCloud<rslidar_ros::Point> pl_orig; // 原始点云
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size(); // 点数
    pl_surf.reserve(plsize);

    static double omega_l = 3.61;

    std::vector<bool> is_first(N_SCANS, true);  // 是否是这一ring的第一个点
    std::vector<double> yaw_fp(N_SCANS, 0.0);   // yaw of first scan point  //这一ring第一个点的yaw
    std::vector<float> yaw_last(N_SCANS, 0.0);  // yaw of last scan point // 这一ring上一个点的yaw
    std::vector<float> time_last(N_SCANS, 0.0); // last offset time// 这一ring上一个点的time
    /*****************************************************************/

    for (int i = 0; i < plsize; i++)
    {
        PointType added_pt;
        // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;

        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = 0; // curvature unit: ms

        if (i % point_filter_num == 0)
        {
            if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind))
            {
                pl_surf.points.push_back(added_pt);
            }
        }
    }

    pub_func(pl_full, pub_full, msg->header.stamp);
    pub_func(pl_surf, pub_surf, msg->header.stamp);
    pub_func(pl_corn, pub_corn, msg->header.stamp);
}

void velo16_handler1(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    double t1 = omp_get_wtime();

    pcl::PointCloud<PointType> pl_full, // 全部的原始点云，use curvature as time of each laser points
        pl_corn,
        pl_surf; // 降采样的原始点云

    pcl::PointCloud<velodyne_ros::Point> pl_orig; // 原始点云
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size(); // 点数
    pl_surf.reserve(plsize);

    static double omega_l = 3.61;

    std::vector<bool> is_first(N_SCANS, true);  // 是否是这一ring的第一个点
    std::vector<double> yaw_fp(N_SCANS, 0.0);   // yaw of first scan point  //这一ring第一个点的yaw
    std::vector<float> yaw_last(N_SCANS, 0.0);  // yaw of last scan point // 这一ring上一个点的yaw
    std::vector<float> time_last(N_SCANS, 0.0); // last offset time// 这一ring上一个点的time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0) // 有时间
    {
        given_offset_time_ = true;
    }
    else // 没有时间
    {
        given_offset_time_ = false;
        double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578; // 第一个点的yaw角，角度制
        double yaw_end = yaw_first;                                                    // 最后一个点的yaw角
        int layer_first = pl_orig.points[0].ring;                                      // 第一个点所在的线ID
        for (uint i = plsize - 1; i > 0; i--)                                          // note 这里是倒排
        {
            if (pl_orig.points[i].ring == layer_first) // 与第一个点共线
            {
                yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
                break;
            }
        }
    }

    for (int i = 0; i < plsize; i++)
    {
        PointType added_pt;
        // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;

        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time / 1000.0; // curvature unit: ms

        if (!given_offset_time_)
        {
            int layer = pl_orig.points[i].ring;
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

            if (is_first[layer])
            {
                // printf("layer: %d; is first: %d", layer, is_first[layer]);
                yaw_fp[layer] = yaw_angle;
                is_first[layer] = false;
                added_pt.curvature = 0.0;
                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.curvature;
                continue;
            }

            // compute offset time
            if (yaw_angle <= yaw_fp[layer])
            {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
            }
            else
            {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
            }

            if (added_pt.curvature < time_last[layer])
                added_pt.curvature += 360.0 / omega_l;

            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.curvature;
        }

        if (i % point_filter_num == 0)
        {
            if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind))
            {
                pl_surf.points.push_back(added_pt);
            }
        }
    }

    pub_func(pl_full, pub_full, msg->header.stamp);
    pub_func(pl_surf, pub_surf, msg->header.stamp);
    pub_func(pl_corn, pub_corn, msg->header.stamp);
}

namespace ouster_ros
{

    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D;
        float intensity;
        uint32_t t;
        uint16_t reflectivity;
        uint8_t ring;
        uint16_t ambient;
        uint32_t range;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
} // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)
// clang-format on

void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    // cout << "oust64_handler" <<endl;
    pcl::PointCloud<PointType> pl_processed;
    pcl::PointCloud<ouster_ros::Point> pl_orig;
    // pcl::PointCloud<pcl::PointXYZI> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    uint plsize = pl_orig.size();

    double time_stamp = msg->header.stamp.toSec();
    pl_processed.clear();
    pl_processed.reserve(pl_orig.points.size());
    for (int i = 0; i < pl_orig.points.size(); i++)
    {
        double range = std::sqrt(pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                                 pl_orig.points[i].z * pl_orig.points[i].z);
        if (range < blind)
        {
            continue;
        }
        Eigen::Vector3d pt_vec;
        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        double yaw_angle = std::atan2(added_pt.y, added_pt.x) * 57.3;
        if (yaw_angle >= 180.0)
            yaw_angle -= 360.0;
        if (yaw_angle <= -180.0)
            yaw_angle += 360.0;

        added_pt.curvature = (pl_orig.points[i].t / 1e9) * 1000.0;

        pl_processed.points.push_back(added_pt);
        if (0) // For debug
        {
            if (pl_processed.size() % 1000 == 0)
            {
                printf("[%d] (%.2f, %.2f, %.2f), ( %.2f, %.2f, %.2f ) | %.2f | %.3f,  \r\n", i, pl_orig.points[i].x, pl_orig.points[i].y,
                       pl_orig.points[i].z, pl_processed.points.back().normal_x, pl_processed.points.back().normal_y,
                       pl_processed.points.back().normal_z, yaw_angle, pl_processed.points.back().intensity);
                // printf("(%d, %.2f, %.2f)\r\n", pl_orig.points[i].ring, pl_orig.points[i].t, pl_orig.points[i].range);
                // printf("(%d, %d, %d)\r\n", pl_orig.points[i].ring, 1, 2);
                cout << (int)(pl_orig.points[i].ring) << ", " << (pl_orig.points[i].t / 1e9) << ", " << pl_orig.points[i].range << endl;
            }
        }
    }
    pub_func(pl_processed, pub_full, msg->header.stamp);
    pub_func(pl_processed, pub_surf, msg->header.stamp);
    pub_func(pl_processed, pub_corn, msg->header.stamp);
}

/**
 * @brief   提取特征点
 *      g_if_using_raw_point == 1 使用的原始点云 pl_surf = pl， pl_corn = 空
 *
 * @param pl     按scan分的原始点云
 * @param types
 * @param pl_corn  特征点:角点
 * @param pl_surf  特征点:面点
 */
void give_feature(pcl::PointCloud<PointType> &pl, vector<orgtype> &types, pcl::PointCloud<PointType> &pl_corn,
                  pcl::PointCloud<PointType> &pl_surf)
{
    uint plsize = pl.size();
    uint plsize2;
    if (plsize == 0)
    {
        printf("something wrong\n");
        return;
    }

    // 跳过盲区点云
    uint head = 0;
    while (types[head].range < blind)
    {
        head++;
    }

    // Surf
    plsize2 = (plsize > group_size) ? (plsize - group_size) : 0;

    Eigen::Vector3d curr_direct(Eigen::Vector3d::Zero());
    Eigen::Vector3d last_direct(Eigen::Vector3d::Zero());

    uint i_nex = 0, i2;
    uint last_i = 0;
    uint last_i_nex = 0;
    int last_state = 0;
    int plane_type;

    PointType ap;
    for (uint i = head; i < plsize2; i += g_LiDAR_sampling_point_step)
    {
        if (types[i].range > blind)
        {
            ap.x = pl[i].x;
            ap.y = pl[i].y;
            ap.z = pl[i].z;
            ap.curvature = pl[i].curvature; // 曲率里面存的是时间
            pl_surf.push_back(ap);
        }
        if (g_if_using_raw_point)
        {
            continue;
        }
        // i_nex = i;
        i2 = i;
        // std::cout<<" i: "<<i<<" i_nex "<<i_nex<<"group_size: "<<group_size<<" plsize "<<plsize<<" plsize2
        // "<<plsize2<<std::endl;
        plane_type = plane_judge(pl, types, i, i_nex, curr_direct);

        if (plane_type == 1)
        {
            for (uint j = i; j <= i_nex; j++)
            {
                if (j != i && j != i_nex) // 不在群的首尾
                {
                    types[j].ftype = Real_Plane;
                }
                else
                {
                    types[j].ftype = Poss_Plane;
                }
            }

            // if(last_state==1 && fabs(last_direct.sum())>0.5)
            // last_state == 1上一个群是平面 && last_direct.norm() > 0.1 表示有上一次的方向
            if (last_state == 1 && last_direct.norm() > 0.1) //
            {
                double mod = last_direct.transpose() * curr_direct; // 夹角
                if (mod > -0.707 && mod < 0.707)                    // 夹角大于45°
                {
                    types[i].ftype = Edge_Plane;
                }
                else
                {
                    types[i].ftype = Real_Plane;
                }
            }

            i = i_nex - 1;
            last_state = 1;
        }
        else if (plane_type == 2) // 盲区点
        {
            i = i_nex;
            last_state = 0;
        }
        else if (plane_type == 0)
        {
            if (last_state == 1)
            {
                uint i_nex_tem;
                uint j;
                for (j = last_i + 1; j <= last_i_nex; j++)
                {
                    uint i_nex_tem2 = i_nex_tem;
                    Eigen::Vector3d curr_direct2;

                    uint ttem = plane_judge(pl, types, j, i_nex_tem, curr_direct2);

                    if (ttem != 1)
                    {
                        i_nex_tem = i_nex_tem2;
                        break;
                    }
                    curr_direct = curr_direct2;
                }

                if (j == last_i + 1)
                {
                    last_state = 0;
                }
                else
                {
                    for (uint k = last_i_nex; k <= i_nex_tem; k++)
                    {
                        if (k != i_nex_tem)
                        {
                            types[k].ftype = Real_Plane;
                        }
                        else
                        {
                            types[k].ftype = Poss_Plane;
                        }
                    }
                    i = i_nex_tem - 1;
                    i_nex = i_nex_tem;
                    i2 = j - 1;
                    last_state = 1;
                }
            }
        }

        last_i = i2;
        last_i_nex = i_nex;
        last_direct = curr_direct;
    }
    if (g_if_using_raw_point)
    {
        return;
    }
    plsize2 = plsize > 3 ? plsize - 3 : 0;
    for (uint i = head + 3; i < plsize2; i++)
    {
        if (types[i].range < blind || types[i].ftype >= Real_Plane)
        {
            continue;
        }

        if (types[i - 1].dista < 1e-16 || types[i].dista < 1e-16)
        {
            continue;
        }

        Eigen::Vector3d vec_a(pl[i].x, pl[i].y, pl[i].z);
        Eigen::Vector3d vecs[2];

        for (int j = 0; j < 2; j++)
        {
            int m = -1;
            if (j == 1)
            {
                m = 1;
            }

            if (types[i + m].range < blind)
            {
                if (types[i].range > inf_bound)
                {
                    types[i].edj[j] = Nr_inf;
                }
                else
                {
                    types[i].edj[j] = Nr_blind;
                }
                continue;
            }

            vecs[j] = Eigen::Vector3d(pl[i + m].x, pl[i + m].y, pl[i + m].z);
            vecs[j] = vecs[j] - vec_a;

            types[i].angle[j] = vec_a.dot(vecs[j]) / vec_a.norm() / vecs[j].norm();
            if (types[i].angle[j] < jump_up_limit)
            {
                types[i].edj[j] = Nr_180;
            }
            else if (types[i].angle[j] > jump_down_limit)
            {
                types[i].edj[j] = Nr_zero;
            }
        }

        types[i].intersect = vecs[Prev].dot(vecs[Next]) / vecs[Prev].norm() / vecs[Next].norm();
        if (types[i].edj[Prev] == Nr_nor && types[i].edj[Next] == Nr_zero && types[i].dista > 0.0225 &&
            types[i].dista > 4 * types[i - 1].dista)
        {
            if (types[i].intersect > cos160)
            {
                if (edge_jump_judge(pl, types, i, Prev))
                {
                    types[i].ftype = Edge_Jump;
                }
            }
        }
        else if (types[i].edj[Prev] == Nr_zero && types[i].edj[Next] == Nr_nor && types[i - 1].dista > 0.0225 &&
                 types[i - 1].dista > 4 * types[i].dista)
        {
            if (types[i].intersect > cos160)
            {
                if (edge_jump_judge(pl, types, i, Next))
                {
                    types[i].ftype = Edge_Jump;
                }
            }
        }
        else if (types[i].edj[Prev] == Nr_nor && types[i].edj[Next] == Nr_inf)
        {
            if (edge_jump_judge(pl, types, i, Prev))
            {
                types[i].ftype = Edge_Jump;
            }
        }
        else if (types[i].edj[Prev] == Nr_inf && types[i].edj[Next] == Nr_nor)
        {
            if (edge_jump_judge(pl, types, i, Next))
            {
                types[i].ftype = Edge_Jump;
            }
        }
        else if (types[i].edj[Prev] > Nr_nor && types[i].edj[Next] > Nr_nor)
        {
            if (types[i].ftype == Nor)
            {
                types[i].ftype = Wire;
            }
        }
    }

    plsize2 = plsize - 1;
    double ratio;
    for (uint i = head + 1; i < plsize2; i++)
    {
        if (types[i].range < blind || types[i - 1].range < blind || types[i + 1].range < blind)
        {
            continue;
        }

        if (types[i - 1].dista < 1e-8 || types[i].dista < 1e-8)
        {
            continue;
        }

        if (types[i].ftype == Nor)
        {
            if (types[i - 1].dista > types[i].dista)
            {
                ratio = types[i - 1].dista / types[i].dista;
            }
            else
            {
                ratio = types[i].dista / types[i - 1].dista;
            }

            if (types[i].intersect < smallp_intersect && ratio < smallp_ratio)
            {
                if (types[i - 1].ftype == Nor)
                {
                    types[i - 1].ftype = Real_Plane;
                }
                if (types[i + 1].ftype == Nor)
                {
                    types[i + 1].ftype = Real_Plane;
                }
                types[i].ftype = Real_Plane;
            }
        }
    }

    int last_surface = -1;
    for (uint j = head; j < plsize; j++)
    {
        if (types[j].ftype == Poss_Plane || types[j].ftype == Real_Plane)
        {
            if (last_surface == -1)
            {
                last_surface = j;
            }

            if (j == uint(last_surface + point_filter_num - 1))
            {
                PointType ap;
                ap.x = pl[j].x;
                ap.y = pl[j].y;
                ap.z = pl[j].z;
                ap.curvature = pl[j].curvature;
                pl_surf.push_back(ap);

                last_surface = -1;
            }
        }
        else
        {
            if (types[j].ftype == Edge_Jump || types[j].ftype == Edge_Plane)
            {
                pl_corn.push_back(pl[j]);
            }
            if (last_surface != -1)
            {
                PointType ap;
                for (uint k = last_surface; k < j; k++)
                {
                    ap.x += pl[k].x;
                    ap.y += pl[k].y;
                    ap.z += pl[k].z;
                    ap.curvature += pl[k].curvature;
                }
                ap.x /= (j - last_surface);
                ap.y /= (j - last_surface);
                ap.z /= (j - last_surface);
                ap.curvature /= (j - last_surface);
                pl_surf.push_back(ap);
            }
            last_surface = -1;
        }
    }
}

void pub_func(pcl::PointCloud<PointType> &pl, ros::Publisher pub, const ros::Time &ct)
{

    pl.height = 1;
    pl.width = pl.size();
    sensor_msgs::PointCloud2 output;

    pcl::toROSMsg(pl, output);

    output.header.frame_id = "livox";
    output.header.stamp = ct;

    pub.publish(output);
}

/**
 * @brief    判断一个点是否是平面点，计算群的方向
 *
 * @param pl            全部点云
 * @param types
 * @param i_cur         当前点索引
 * @param i_nex         同一个群里的下一个点索引（退出时为下一个群的起点）
 * @param curr_direct   群内第一个点指向最后一个点的方向向量
 * @return int   0不是 1 平面  2盲区
 */
int plane_judge(const pcl::PointCloud<PointType> &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct)
{
    // 计算一个群的距离
    double group_dis = disA * types[i_cur].range + disB;
    group_dis = group_dis * group_dis;
    // i_nex = i_cur;

    double two_dis;
    vector<double> disarr; // 当前点和下一个点之间的距离
    disarr.reserve(20);

    for (i_nex = i_cur; i_nex < i_cur + group_size; i_nex++)
    {
        if (types[i_nex].range < blind)
        {
            curr_direct.setZero();
            return 2;
        }
        disarr.push_back(types[i_nex].dista);
    }

    // 无限循环指导距离内的点都加入进一个群
    for (;;)
    {
        if ((i_cur >= pl.size()) || (i_nex >= pl.size()))
            break;

        if (types[i_nex].range < blind)
        {
            curr_direct.setZero();
            return 2;
        }
        vx = pl[i_nex].x - pl[i_cur].x;
        vy = pl[i_nex].y - pl[i_cur].y;
        vz = pl[i_nex].z - pl[i_cur].z;
        two_dis = vx * vx + vy * vy + vz * vz;
        if (two_dis >= group_dis)
        {
            break;
        }
        disarr.push_back(types[i_nex].dista);
        i_nex++;
    }

    // 退出上的循环后，vx,vy,vz代 表群内第一个点指向最后一个点的方向
    double leng_wid = 0; // 存储最大的垂直距离
    double v1[3], v2[3];
    // 遍历群中每一个点
    for (uint j = i_cur + 1; j < i_nex; j++)
    {
        if ((j >= pl.size()) || (i_cur >= pl.size()))
            break;
        v1[0] = pl[j].x - pl[i_cur].x;
        v1[1] = pl[j].y - pl[i_cur].y;
        v1[2] = pl[j].z - pl[i_cur].z;

        // 叉乘
        v2[0] = v1[1] * vz - vy * v1[2];
        v2[1] = v1[2] * vx - v1[0] * vz;
        v2[2] = v1[0] * vy - vx * v1[1];

        double lw = v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2];
        if (lw > leng_wid)
        {
            leng_wid = lw;
        }
    }

    if ((two_dis * two_dis / leng_wid) < p2l_ratio)
    {
        curr_direct.setZero();
        return 0;
    }

    // 从大到小排序
    uint disarrsize = disarr.size();
    for (uint j = 0; j < disarrsize - 1; j++)
    {
        for (uint k = j + 1; k < disarrsize; k++)
        {
            if (disarr[j] < disarr[k])
            {
                leng_wid = disarr[j];
                disarr[j] = disarr[k];
                disarr[k] = leng_wid;
            }
        }
    }

    if (disarr[disarr.size() - 2] < 1e-16)
    {
        curr_direct.setZero();
        return 0;
    }

    if (lidar_type == MID || lidar_type == HORIZON)
    {
        double dismax_mid = disarr[0] / disarr[disarrsize / 2];              // 最大值/中值
        double dismid_min = disarr[disarrsize / 2] / disarr[disarrsize - 2]; // 中值/最小值

        if (dismax_mid >= limit_maxmid || dismid_min >= limit_midmin)
        {
            curr_direct.setZero();
            return 0;
        }
    }
    else
    {
        double dismax_min = disarr[0] / disarr[disarrsize - 2]; // 最大值/最小值
        if (dismax_min >= limit_maxmin)
        {
            curr_direct.setZero();
            return 0;
        }
    }

    curr_direct << vx, vy, vz;
    curr_direct.normalize();
    return 1;
}

bool edge_jump_judge(const pcl::PointCloud<PointType> &pl, vector<orgtype> &types, uint i, Surround nor_dir)
{
    if (nor_dir == 0)
    {
        if (types[i - 1].range < blind || types[i - 2].range < blind)
        {
            return false;
        }
    }
    else if (nor_dir == 1)
    {
        if (types[i + 1].range < blind || types[i + 2].range < blind)
        {
            return false;
        }
    }
    double d1 = types[i + nor_dir - 1].dista;
    double d2 = types[i + 3 * nor_dir - 2].dista;
    double d;

    if (d1 < d2)
    {
        d = d1;
        d1 = d2;
        d2 = d;
    }

    d1 = sqrt(d1);
    d2 = sqrt(d2);

    if (d1 > edgea * d2 || (d1 - d2) > edgeb)
    {
        return false;
    }

    return true;
}
