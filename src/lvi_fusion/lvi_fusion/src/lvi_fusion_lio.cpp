/* 
This code is the implementation of our paper "lvi_fusion: A Robust, Real-time, RGB-colored, 
LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package".

Author: Jiarong Lin   < ziv.lin.ljr@gmail.com >

If you use any code of this repo in your academic research, please cite at least
one of our papers:
[1] Lin, Jiarong, and Fu Zhang. "lvi_fusion: A Robust, Real-time, RGB-colored, 
    LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package." 
[2] Xu, Wei, et al. "Fast-lio2: Fast direct lidar-inertial odometry."
[3] Lin, Jiarong, et al. "R2LIVE: A Robust, Real-time, LiDAR-Inertial-Visual
     tightly-coupled state Estimator and mapping." 
[4] Xu, Wei, and Fu Zhang. "Fast-lio: A fast, robust lidar-inertial odometry 
    package by tightly-coupled iterated kalman filter."
[5] Cai, Yixi, Wei Xu, and Fu Zhang. "ikd-Tree: An Incremental KD Tree for 
    Robotic Applications."
[6] Lin, Jiarong, and Fu Zhang. "Loam-livox: A fast, robust, high-precision 
    LiDAR odometry and mapping package for LiDARs of small FoV."

For commercial use, please contact me < ziv.lin.ljr@gmail.com > and
Dr. Fu Zhang < fuzhang@hku.hk >.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from this
    software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
*/
#include "lvi_fusion.hpp"

// ??? IMU data??????buff
void lvi_fusion::imu_cbk( const sensor_msgs::Imu::ConstPtr &msg_in )
{
    //  cout << "?????????IMU" <<endl;
    sensor_msgs::Imu::Ptr msg( new sensor_msgs::Imu( *msg_in ) );
    double                timestamp = msg->header.stamp.toSec();
    g_camera_lidar_queue.imu_in( timestamp ); // ??????last_timestamp_imu
    mtx_buffer.lock();

    // ??????????????????
    if ( timestamp < last_timestamp_imu )
    {
        ROS_ERROR( "imu loop back, clear buffer" );
        imu_buffer_lio.clear();
        imu_buffer_vio.clear();
        flg_reset = true;
    }

    last_timestamp_imu = timestamp;

    // ??????????????? m/s2
    if ( g_camera_lidar_queue.m_if_acc_mul_G ) 
    {
        msg->linear_acceleration.x *= G_m_s2;
        msg->linear_acceleration.y *= G_m_s2;
        msg->linear_acceleration.z *= G_m_s2;
    }

    imu_buffer_lio.push_back( msg );
    imu_buffer_vio.push_back( msg );

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void printf_field_name( sensor_msgs::PointCloud2::ConstPtr &msg )
{
    cout << "Input pointcloud field names: [" << msg->fields.size() << "]: ";
    for ( size_t i = 0; i < msg->fields.size(); i++ )
    {
        cout << msg->fields[ i ].name << ", ";
    }
    cout << endl;
}

/**
 * @brief ????????????????????????????????????pcl::PointXYZINormal??????
 * 
 * @param msg     lidar_buff??????????????????
 * @param pcl_pc  pcl???????????????
 * @return true   ????????????
 * @return false  ????????????(??????????????????????????????)
 */
bool lvi_fusion::get_pointcloud_data_from_ros_message( sensor_msgs::PointCloud2::ConstPtr &msg, pcl::PointCloud< pcl::PointXYZINormal > &pcl_pc )
{

    // printf("Frame [%d] %.3f ", g_LiDAR_frame_index,  msg->header.stamp.toSec() - g_camera_lidar_queue.m_first_imu_time);
    pcl::PointCloud< pcl::PointXYZI > res_pc;
    scope_color( ANSI_COLOR_YELLOW_BOLD );
    // printf_field_name(msg);
    if ( msg->fields.size() < 3 ) // ??????fields?????????
    {
        cout << "Get pointcloud data from ros messages fail!!!" << endl;
        scope_color( ANSI_COLOR_RED_BOLD );
        printf_field_name( msg );
        return false;
    }
    else
    {   
        /*????????????????????????????????????pcl::PointXYZINormal??????*/
        if ( ( msg->fields.size() == 8 ) && ( msg->fields[ 3 ].name == "intensity" ) &&
             ( msg->fields[ 4 ].name == "normal_x" ) ) // Input message type is pcl::PointXYZINormal
        {
            pcl::fromROSMsg( *msg, pcl_pc );
            return true;
        }
        else if ( ( msg->fields.size() == 4 ) && ( msg->fields[ 3 ].name == "rgb" ) )
        {
            double maximum_range = 5;
            get_ros_parameter< double >( m_ros_node_handle, "iros_range", maximum_range, 5 );
            pcl::PointCloud< pcl::PointXYZRGB > pcl_rgb_pc;
            pcl::fromROSMsg( *msg, pcl_rgb_pc );
            double lidar_point_time = msg->header.stamp.toSec();
            int    pt_count = 0;
            pcl_pc.resize( pcl_rgb_pc.points.size() );
            for ( int i = 0; i < pcl_rgb_pc.size(); i++ )
            {
                pcl::PointXYZINormal temp_pt;
                temp_pt.x = pcl_rgb_pc.points[ i ].x;
                temp_pt.y = pcl_rgb_pc.points[ i ].y;
                temp_pt.z = pcl_rgb_pc.points[ i ].z;
                double frame_dis = sqrt( temp_pt.x * temp_pt.x + temp_pt.y * temp_pt.y + temp_pt.z * temp_pt.z );
                if ( frame_dis > maximum_range )
                {
                    continue;
                }
                temp_pt.intensity = ( pcl_rgb_pc.points[ i ].r + pcl_rgb_pc.points[ i ].g + pcl_rgb_pc.points[ i ].b ) / 3.0;
                temp_pt.curvature = 0;
                pcl_pc.points[ pt_count ] = temp_pt;
                pt_count++;
            }
            pcl_pc.points.resize( pt_count );
            return true;
        }
        else // TODO, can add by yourself
        {
            cout << "Get pointcloud data from ros messages fail!!! ";
            scope_color( ANSI_COLOR_RED_BOLD );
            printf_field_name( msg );
            return false;
        }
    }
}

// ??????????????????????????????IMU??????(?????????????????????scan??????????????????)
bool lvi_fusion::sync_packages( MeasureGroup &meas )
{
    if ( lidar_buffer.empty() || imu_buffer_lio.empty() )
    {
        return false;
    }
    
    /*** push lidar frame ***/
    if ( !lidar_pushed ) // ????????????if()???????????????????????????????????????????????????????????????
    {
        meas.lidar.reset( new PointCloudXYZINormal() );
        if ( get_pointcloud_data_from_ros_message( lidar_buffer.front(), *( meas.lidar ) ) == false ) // ?????????????????????ros-???pcl
        {
            return false;
        }
        // pcl::fromROSMsg(*(lidar_buffer.front()), *(meas.lidar));
        // lidar SCAN??????????????????????????????
        meas.lidar_beg_time = lidar_buffer.front()->header.stamp.toSec();
        lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double( 1000 );
        meas.lidar_end_time = lidar_end_time;
        // printf("Input LiDAR time = %.3f, %.3f\n", meas.lidar_beg_time, meas.lidar_end_time);
        // printf_line_mem_MB;
        lidar_pushed = true;
    }    


    if ( last_timestamp_imu < lidar_end_time ) // ?????????imu????????????scan???????????????
    {
        // cout << setprecision(11)<<"last_timestamp_imu:" << last_timestamp_imu <<endl;
        // cout << setprecision(11)<<"lidar_end_time:" << lidar_end_time <<endl;
        return false;
    }
   
    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer_lio.front()->header.stamp.toSec();
    meas.imu.clear();
    // ?????????scan????????????scan?????????imu_buff????????????meas.imu
    while ( ( !imu_buffer_lio.empty() ) && ( imu_time < lidar_end_time ) )
    {
        imu_time = imu_buffer_lio.front()->header.stamp.toSec();
        if ( imu_time > lidar_end_time + m_imu_interval ) // 0.02 = 50hz TODO ??????????????????IMU??????????????????
            break;
        meas.imu.push_back( imu_buffer_lio.front() );
        imu_buffer_lio.pop_front();
    }
   

    while ( !imu_buffer_vio.empty() ) //????????????????????????cam?????????0.2????????????
    {
        double imu_time = imu_buffer_vio.front()->header.stamp.toSec();
        if ( imu_time > lidar_end_time + m_imu_interval ) 
        {
            break;
        }
        imu_buffer_vio.pop_front();
    }
   
    lidar_buffer.pop_front();
    lidar_pushed = false;
    // if (meas.imu.empty()) return false;
    // std::cout<<"[IMU Sycned]: "<<imu_time<<" "<<lidar_end_time<<std::endl;
    return true;
}

// project lidar frame to world
void lvi_fusion::pointBodyToWorld( PointType const *const pi, PointType *const po )
{
    Eigen::Vector3d p_body( pi->x, pi->y, pi->z );
    Eigen::Vector3d p_global( g_lio_state.rot_end * ( p_body + Lidar_offset_to_IMU ) + g_lio_state.pos_end );

    po->x = p_global( 0 );
    po->y = p_global( 1 );
    po->z = p_global( 2 );
    po->intensity = pi->intensity;
}

// ?????????????????????????????????RGB?????????????????????
void lvi_fusion::RGBpointBodyToWorld( PointType const *const pi, pcl::PointXYZI *const po )
{
    Eigen::Vector3d p_body( pi->x, pi->y, pi->z );
    Eigen::Vector3d p_global( g_lio_state.rot_end * ( p_body + Lidar_offset_to_IMU ) + g_lio_state.pos_end );

    po->x = p_global( 0 );
    po->y = p_global( 1 );
    po->z = p_global( 2 );
    po->intensity = pi->intensity;

    // ??????????????????
    float intensity = pi->intensity;
    intensity = intensity - std::floor( intensity );

    int reflection_map = intensity * 10000; //????????????????????? ??????????????????*10000
}

// 3????????????1?????????
int lvi_fusion::get_cube_index( const int &i, const int &j, const int &k )
{
    return ( i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k );
}

bool lvi_fusion::center_in_FOV( Eigen::Vector3f cube_p )
{
    Eigen::Vector3f dis_vec = g_lio_state.pos_end.cast< float >() - cube_p;
    float           squaredSide1 = dis_vec.transpose() * dis_vec;

    if ( squaredSide1 < 0.4 * cube_len * cube_len )
        return true;

    dis_vec = XAxisPoint_world.cast< float >() - cube_p;
    float squaredSide2 = dis_vec.transpose() * dis_vec;

    float ang_cos =
        fabs( squaredSide1 <= 3 ) ? 1.0 : ( LIDAR_SP_LEN * LIDAR_SP_LEN + squaredSide1 - squaredSide2 ) / ( 2 * LIDAR_SP_LEN * sqrt( squaredSide1 ) );

    return ( ( ang_cos > HALF_FOV_COS ) ? true : false );
}

bool lvi_fusion::if_corner_in_FOV( Eigen::Vector3f cube_p )
{
    Eigen::Vector3f dis_vec = g_lio_state.pos_end.cast< float >() - cube_p;
    float           squaredSide1 = dis_vec.transpose() * dis_vec;
    dis_vec = XAxisPoint_world.cast< float >() - cube_p;
    float squaredSide2 = dis_vec.transpose() * dis_vec;
    float ang_cos =
        fabs( squaredSide1 <= 3 ) ? 1.0 : ( LIDAR_SP_LEN * LIDAR_SP_LEN + squaredSide1 - squaredSide2 ) / ( 2 * LIDAR_SP_LEN * sqrt( squaredSide1 ) );
    return ( ( ang_cos > HALF_FOV_COS ) ? true : false );
}


// LOAM????????????????????????local map
void lvi_fusion::lasermap_fov_segment()
{
    laserCloudValidNum = 0;

    // ??????????????????
    pointBodyToWorld( XAxisPoint_body, XAxisPoint_world );
    // ????????????????????????
    int centerCubeI = int( ( g_lio_state.pos_end( 0 ) + 0.5 * cube_len ) / cube_len ) + laserCloudCenWidth;
    int centerCubeJ = int( ( g_lio_state.pos_end( 1 ) + 0.5 * cube_len ) / cube_len ) + laserCloudCenHeight;
    int centerCubeK = int( ( g_lio_state.pos_end( 2 ) + 0.5 * cube_len ) / cube_len ) + laserCloudCenDepth;
    if ( g_lio_state.pos_end( 0 ) + 0.5 * cube_len < 0 )
        centerCubeI--;
    if ( g_lio_state.pos_end( 1 ) + 0.5 * cube_len < 0 )
        centerCubeJ--;
    if ( g_lio_state.pos_end( 2 ) + 0.5 * cube_len < 0 )
        centerCubeK--;
    bool last_inFOV_flag = 0;
    int  cube_index = 0; // ?????????0
    cub_needrm.clear();
    cub_needad.clear();
    T2[ time_log_counter ] = Measures.lidar_beg_time;
    double t_begin = omp_get_wtime();

    // ????????????????????????I???????????????
    while ( centerCubeI < FOV_RANGE + 1 )
    {
        for ( int j = 0; j < laserCloudHeight; j++ )
        {
            for ( int k = 0; k < laserCloudDepth; k++ )
            {
                int i = laserCloudWidth - 1;

              
                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];  // ???????????????
                last_inFOV_flag = _last_inFOV[ cube_index ];

                // ???????????????I?????????????????????????????????????????????
                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i - 1, j, k ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i - 1, j, k ) ];
                }
                // TODO ?????????????????????????????????????????????????????????
                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }
        centerCubeI++;
        laserCloudCenWidth++;
    }

    // ????????????????????????I???????????????
    while ( centerCubeI >= laserCloudWidth - ( FOV_RANGE + 1 ) )
    {
        for ( int j = 0; j < laserCloudHeight; j++ )
        {
            for ( int k = 0; k < laserCloudDepth; k++ )
            {
                int i = 0;

                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i + 1, j, k ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i + 1, j, k ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeI--;
        laserCloudCenWidth--;
    }

    // ??????
    while ( centerCubeJ < ( FOV_RANGE + 1 ) )
    {
        for ( int i = 0; i < laserCloudWidth; i++ )
        {
            for ( int k = 0; k < laserCloudDepth; k++ )
            {
                int j = laserCloudHeight - 1;

                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i, j - 1, k ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i, j - 1, k ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeJ++;
        laserCloudCenHeight++;
    }

    while ( centerCubeJ >= laserCloudHeight - ( FOV_RANGE + 1 ) )
    {
        for ( int i = 0; i < laserCloudWidth; i++ )
        {
            for ( int k = 0; k < laserCloudDepth; k++ )
            {
                int                       j = 0;
                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i, j + 1, k ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i, j + 1, k ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeJ--;
        laserCloudCenHeight--;
    }

    while ( centerCubeK < ( FOV_RANGE + 1 ) )
    {
        for ( int i = 0; i < laserCloudWidth; i++ )
        {
            for ( int j = 0; j < laserCloudHeight; j++ )
            {
                int                       k = laserCloudDepth - 1;
                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i, j, k - 1 ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i, j, k - 1 ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeK++;
        laserCloudCenDepth++;
    }

    while ( centerCubeK >= laserCloudDepth - ( FOV_RANGE + 1 ) )
    {
        for ( int i = 0; i < laserCloudWidth; i++ )
        {
            for ( int j = 0; j < laserCloudHeight; j++ )
            {
                int                       k = 0;
                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i, j, k + 1 ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i, j, k + 1 ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }
        centerCubeK--;
        laserCloudCenDepth--;
    }

    cube_points_add->clear();
    featsFromMap->clear();
    memset( now_inFOV, 0, sizeof( now_inFOV ) );
    copy_time = omp_get_wtime() - t_begin;
    double fov_check_begin = omp_get_wtime();

    fov_check_time = omp_get_wtime() - fov_check_begin;

    double readd_begin = omp_get_wtime();
#ifdef USE_ikdtree
    if ( cub_needrm.size() > 0 ) // ?????????0????????????
        ikdtree.Delete_Point_Boxes( cub_needrm );
    delete_box_time = omp_get_wtime() - readd_begin;
    // s_plot4.push_back(omp_get_wtime() - t_begin); t_begin = omp_get_wtime();
    if ( cub_needad.size() > 0 )  // ?????????0????????????
        ikdtree.Add_Point_Boxes( cub_needad );
    readd_box_time = omp_get_wtime() - readd_begin - delete_box_time;
    // s_plot5.push_back(omp_get_wtime() - t_begin); t_begin = omp_get_wtime();
    if ( cube_points_add->points.size() > 0 )  // ?????????0????????????
        ikdtree.Add_Points( cube_points_add->points, true );
#endif
    readd_time = omp_get_wtime() - readd_begin - delete_box_time - readd_box_time;
    // s_plot6.push_back(omp_get_wtime() - t_begin);    
}

// ??? Point cloud ???lidar_buff
void lvi_fusion::feat_points_cbk( const sensor_msgs::PointCloud2::ConstPtr &msg_in )
{
    sensor_msgs::PointCloud2::Ptr msg( new sensor_msgs::PointCloud2( *msg_in ) );
    msg->header.stamp = ros::Time( msg_in->header.stamp.toSec() - m_lidar_imu_time_delay );
    if ( g_camera_lidar_queue.lidar_in( msg_in->header.stamp.toSec() + 0.1 ) == 0 ) // ????????????
    {
        return;
    }
    mtx_buffer.lock();

    // ??????????????????
    if ( msg->header.stamp.toSec() < last_timestamp_lidar )
    {
        ROS_ERROR( "lidar loop back, clear buffer" );
        lidar_buffer.clear();

    }
    // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
    lidar_buffer.push_back( msg );
    last_timestamp_lidar = msg->header.stamp.toSec();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// ??? Point cloud ???lidar_buff
void lvi_fusion::feat_points_cbk_corn( const sensor_msgs::PointCloud2::ConstPtr &msg_in )
{
    sensor_msgs::PointCloud2::Ptr msg( new sensor_msgs::PointCloud2( *msg_in ) );
    msg->header.stamp = ros::Time( msg_in->header.stamp.toSec() - m_lidar_imu_time_delay );
  
    mtx_buffer.lock();
  
    if ( msg->header.stamp.toSec() < last_timestamp_lidar )
    {
        ROS_ERROR( "lidar loop back, clear buffer" );
        lidar_buffer_corn.clear();
    }
    // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
    lidar_buffer_corn.push_back( msg );
    // last_timestamp_lidar = msg->header.stamp.toSec();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}


void lvi_fusion::wait_render_thread_finish()
{
    if ( m_render_thread != nullptr )
    {
        m_render_thread->get(); // wait render thread to finish.
        // m_render_thread = nullptr;
    }
}


void lvi_fusion::vector2double(const std::shared_ptr<std::list<Frame>> lidarFrameList){
  // int Frame_size = lidarFrameList->size();
  int i = 0;
  for(const auto& l : *lidarFrameList){
    Eigen::Map<Eigen::Matrix<double, 7, 1>> PR(para_PR[i]); // ????????????????????????????????????PR???para_PR????????????
    PR.segment<3>(0) = l.P;
    PR.segment<4>(3) = l.Q.coeffs();

    Eigen::Map<Eigen::Matrix<double, 9, 1>> VBias(para_VBias[i]);
    VBias.segment<3>(0) = l.V;
    VBias.segment<3>(3) = l.ba;
    VBias.segment<3>(6) = l.bg;
    i++;
  }
  
}

void lvi_fusion::double2vector(std::shared_ptr<std::list<Frame>> lidarFrameList){
  int i = 0;
  for(auto& l : *lidarFrameList){
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> PR(para_PR[i]);
    l.P = PR.segment<3>(0);
    l.Q =PR.segment<4>(3);
    l.Q.normalize();

    Eigen::Map<const Eigen::Matrix<double, 9, 1>> VBias(para_VBias[i]);
   
    l.V = VBias.segment<3>(0);
    l.ba = VBias.segment<3>(3);
    l.bg = VBias.segment<3>(6);
    // ??????????????????
    // if(l.ba.norm()>1.0) l.ba =Eigen::Vector3d::Zero();
    // if(l.bg.norm()>1.0) l.bg =Eigen::Vector3d::Zero();
    i++;
  }

}


// ????????????
void lvi_fusion::processPointToPlan(std::vector<ceres::CostFunction *>& edges,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
                                   const Eigen::Matrix4d& m4d){  // Twl


  PointType  _pointSel, _coeff;
  std::vector<float> _pointSearchSqDis;
  


  // Eigen::Matrix< double, 3, 1 > _matX0;
  // _matX0.setZero();
  Eigen::Vector3d norm;
  norm.setZero();
  int laserCloudSurfStackNum = laserCloudSurf->points.size();

    std::vector< bool >               point_selected_surf( laserCloudSurfStackNum, true );  // ??????????????????????????????????????????
    std::vector< PointVector >        Nearest_Points( laserCloudSurfStackNum );  // ??????????????????????????????????????????????????????????????????????????????????????????kdtree???????????????

  double maximum_pt_range = 0.0;
//   cout << "laserCloudSurfStackNum"<< laserCloudSurfStackNum<<endl;
  int debug_num = 0;
  int debug_num2 = 0;
  int debug_num3 = 0;
  for (int i = 0; i < laserCloudSurfStackNum; i++) {
    PointType &_pointOri = laserCloudSurf->points[i];
    double     ori_pt_dis = sqrt( _pointOri.x * _pointOri.x + _pointOri.y * _pointOri.y + _pointOri.z * _pointOri.z );                 
    maximum_pt_range = std::max( ori_pt_dis, maximum_pt_range );
    pointAssociateToMap(&_pointOri, &_pointSel, m4d);

    if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) continue;

    auto &points_near = Nearest_Points[ i ];

    if(ikdtree.size() > 50) {
        
 
      ikdtree.Nearest_Search( _pointSel, NUM_MATCH_POINTS, points_near, _pointSearchSqDis );
      if ( _pointSearchSqDis[NUM_MATCH_POINTS-1] > m_maximum_pt_kdtree_dis )
      {
        point_selected_surf[ i ] = false;
        debug_num++;
      }
    
      if ( point_selected_surf[ i ] == false )
        continue;
      // ?????????????????????d???1
   
        cv::Mat matA0( NUM_MATCH_POINTS, 3, CV_32F, cv::Scalar::all( 0 ) );
        cv::Mat matB0( NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all( -1 ) );
        cv::Mat matX0( NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all( 0 ) );
        for (int j = 0; j < NUM_MATCH_POINTS; j++) {
            matA0.at< float >( j, 0 ) = points_near[ j ].x;
            matA0.at< float >( j, 1 ) = points_near[ j ].y;
            matA0.at< float >( j, 2 ) = points_near[ j ].z;
        }
      
        cv::solve( matA0, matB0, matX0, cv::DECOMP_QR ); // TODO
        //   pa /= ps;
        // pb /= ps;
        // pc /= ps;
        // pd /= ps;
        float pa = matX0.at< float >( 0, 0 );
        float pb = matX0.at< float >( 1, 0 );
        float pc = matX0.at< float >( 2, 0 );
        float pd = 1;
        float ps = sqrt( pa * pa + pb * pb + pc * pc );
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;
   

          // ????????????????????????????????????????????????
        bool planeValid = true;
        for (int j = 0; j < NUM_MATCH_POINTS; j++) {
            if (fabs( pa * points_near[ j ].x + pb * points_near[ j ].y + pc * points_near[ j ].z + pd ) >m_planar_check_dis)  
            {
                planeValid = false;
                point_selected_surf[ i ] = false;
                debug_num2++;
                break;
            }
        }

        if (planeValid) {
       
          float pd2 = pa * _pointSel.x + pb * _pointSel.y + pc * _pointSel.z + pd;
          Eigen::Vector3d point_proj = Eigen::Vector3d(_pointSel.x,_pointSel.y,_pointSel.z) - (pd2 * Eigen::Vector3d(pa,pb,pc));
          double acc_distance = ( ori_pt_dis < m_long_rang_pt_dis ) ? m_maximum_res_dis : 1.0;
          if ( pd2 < acc_distance )
          {
           Eigen::Matrix4d Tbl;
            Tbl.topLeftCorner(3,3) =Eigen::Matrix3d::Identity();// TODO ?????????????????????R=1
            Tbl.topRightCorner(3,1) = Lidar_offset_to_IMU;
            Eigen::Vector3d e1(1, 0, 0);
            Eigen::Matrix3d J = e1 * norm.transpose();
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::Matrix3d R_svd = svd.matrixV() * svd.matrixU().transpose();
            Eigen::Matrix3d info = 1/1.5 *Eigen::Matrix3d::Identity();
            // info(1, 1) *= 0.03;
            // info(2, 2) *= 0.03;
            Eigen::Matrix3d sqrt_info = info * R_svd.transpose();
            // cout << " R_svd.transpose()" <<  R_svd.transpose()<<endl;
            // cout << " sqrt_info" <<  sqrt_info<<endl;
            // auto* e = Cost_NavState_IMU_Plan_Vec::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
            //                                            point_proj,
            //                                            Tbl,
            //                                            info);
            auto *e = LidarPlaneNormFactor::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z), Eigen::Vector3d(pa,pb,pc), pd,Lidar_offset_to_IMU,Eigen::Matrix3d::Identity());
            edges.push_back(e);
          }
          else{
               point_selected_surf[ i ] = false;
               debug_num3++;
          }
      
        }
        
  
    }
    else{
        cout << "kdtree don???t have enough points" <<endl;
    }

  }
//   cout << "debug:"<< debug_num << ","<< debug_num2 <<","<<debug_num3<<endl;

}

void lvi_fusion::ALoamEstimate(std::shared_ptr<std::list<Frame>> lidarFrameList)
{
  Eigen::Matrix3d exRbl =  Eigen::Matrix3d::Identity();
  auto exPbl = Lidar_offset_to_IMU;
  int Frame_size = lidarFrameList->size();
  int stack_count = 0;
  for(const auto& l : (*lidarFrameList)){
    // laserCloudSurfStack[stack_count]->clear(); // ??????clear??????lidarFrameList??????????????????
    laserCloudSurfStack[stack_count] = l.SurfaceCloud;

    stack_count++;
  }


  Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
  // if(Frame_size<2) return;

  PointType pointOri, pointSel;



  
  // ???????????????????????????????????????????????????????????????
  for (int iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++)
  {
    
    vector2double(lidarFrameList);


    ceres::LossFunction *Lidar_loss_function = nullptr;
    // ceres::LossFunction *Lidar_loss_function = new ceres::HuberLoss(0.1);



    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);



    for (int i = 0; i < Frame_size; i++)
    {
      ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
      problem.AddParameterBlock(para_PR[i], SIZE_POSE, local_parameterization);
      problem.AddParameterBlock(para_VBias[i], SIZE_SPEEDBIAS);
      
    }
   


        // ???????????????????????????
    for(int f=1; f<Frame_size; ++f){
      auto frame_curr = lidarFrameList->begin();
      std::advance(frame_curr, f);
      // cout << "??????imu??????:"<< f<<endl;
      IMUFactor* imu_factor = new IMUFactor(frame_curr->imuIntegrator);
      problem.AddResidualBlock(imu_factor, nullptr, para_PR[f-1], para_VBias[f-1], para_PR[f],para_VBias[f]);

    }

    if (last_marginalization_info){
      // construct new marginlization_factor
      MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
      problem.AddResidualBlock(marginalization_factor, nullptr,
                               last_marginalization_parameter_blocks);
    }
    


    // ????????????q???t
    Eigen::Quaterniond q_before_opti = lidarFrameList->back().Q;
    Eigen::Vector3d t_before_opti = lidarFrameList->back().P;

    // ceres????????????
 
    std::vector<std::vector<ceres::CostFunction *>> edgesPlan(Frame_size);



    // ?????????????????????????????????
    // if(iterCount%2==0){

        for(int f=0; f<Frame_size; ++f) {
            auto frame_curr = lidarFrameList->begin();
            std::advance(frame_curr, f);
            transformTobeMapped = Eigen::Matrix4d::Identity();
            transformTobeMapped.topLeftCorner(3,3) = frame_curr->Q * exRbl;// TODO ?????????????????????R=1
            transformTobeMapped.topRightCorner(3,1) = frame_curr->Q * exPbl + frame_curr->P;

            processPointToPlan(edgesPlan[f],laserCloudSurfStack[f],transformTobeMapped); // ????????????frame??? surf_??????
        }
    // }

    //   cout <<"edgesPlan[0]" <<edgesPlan[0].size()<< endl;
  
  
      // if(iterCount == 0){ // ?????????????????????????????????processPointToLine??????????????????????????????????????????????????????????????????????????????????????????????????????
    for(int f=0; f<Frame_size; ++f){
      for (auto &e : edgesPlan[f]) {
          
          problem.AddResidualBlock(e, Lidar_loss_function, para_PR[f]);

      }
    }
    
 
    
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.linear_solver_type = ceres::DENSE_QR;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    // options.max_num_iterations = 4;
    // options.minimizer_progress_to_stdout = false;
    // options.check_gradients = false;
	// options.gradient_check_relative_precision = 1e-4;
    // options.num_threads = 1;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    double2vector(lidarFrameList); // ?????????????????????


    Eigen::Quaterniond q_after_opti = lidarFrameList->back().Q;
    Eigen::Vector3d t_after_opti = lidarFrameList->back().P;
    Eigen::Vector3d V_after_opti = lidarFrameList->back().V;
    double deltaR = (q_before_opti.angularDistance(q_after_opti)) * 180.0 / M_PI;
    double deltaT = (t_before_opti - t_after_opti).norm();
  
    // ????????????????????? || ??????????????????????????????
    if (deltaR < 0.01 && deltaT< 0.015 || (iterCount+1) == NUM_MAX_ITERATIONS){

    //    ROS_INFO("Frame: %d\n",frame_count++);
      if(Frame_size != Lidar_WINDOW_SIZE) break;
 
      // apply marginalization ???VINS?????????old??????${mySRC_LIST}
     
      
      auto *marginalization_info = new MarginalizationInfo();

      if (last_marginalization_info){ // ??????????????????????????????
        std::vector<int> drop_set;
         // last_marginalization_parameter_blocks??????????????????????????????????????????????????????
        for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
        {
          // ???????????????????????????????????????????????????????????????????????????????????????????????????
          if (last_marginalization_parameter_blocks[i] == para_PR[0] ||
              last_marginalization_parameter_blocks[i] == para_VBias[0])
            drop_set.push_back(i);
        }

        auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        auto *residual_block_info = new ResidualBlockInfo(marginalization_factor, nullptr,
                                                          last_marginalization_parameter_blocks,
                                                          drop_set);
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
      
      
     
      
    //   auto frame_curr = lidarFrameList->begin();
    //   // TODO ?????????????????????????????????
    //   std::advance(frame_curr, 1);// ?????????
      // ????????????????????????????????????
     

    auto frame_curr = lidarFrameList->begin();
    std::advance(frame_curr, 1);// ?????????
    IMUFactor* imu_factor = new IMUFactor(frame_curr->imuIntegrator);
    auto *residual_block_info = new ResidualBlockInfo(imu_factor, nullptr,
                                                        std::vector<double *>{para_PR[0], para_VBias[0], para_PR[1], para_VBias[1]},
                                                        std::vector<int>{0, 1}); // ????????????????????????????????????????????????0???1????????????????????????????????????
    marginalization_info->addResidualBlockInfo(residual_block_info);
       


     
      // ?????????????????????
      int f = 0;
      transformTobeMapped = Eigen::Matrix4d::Identity();
      transformTobeMapped.topLeftCorner(3,3) = frame_curr->Q * exRbl;
      transformTobeMapped.topRightCorner(3,1) = frame_curr->Q * exPbl + frame_curr->P;
    //   edgesLine[f].clear();
      edgesPlan[f].clear();
         processPointToPlan(edgesPlan[f],laserCloudSurfStack[f],transformTobeMapped); // ????????????frame??? surf_??????
        

      for (auto &e : edgesPlan[f]) {
     
            auto *residual_block_info = new ResidualBlockInfo(e, nullptr,
                                                              std::vector<double *>{para_PR[0]},
                                                              std::vector<int>{0});
            marginalization_info->addResidualBlockInfo(residual_block_info);
   
      }



      marginalization_info->preMarginalize();
      // cout<<"pre marginalization"<<endl;
      marginalization_info->marginalize();
      // cout<<"marginalize"<<endl;
      // ??????????????????????????????????????????????????????
      std::unordered_map<long, double *> addr_shift;

      // ??????????????????
      for (int i = 1; i < Lidar_WINDOW_SIZE; i++)
      {
        addr_shift[reinterpret_cast<long>(para_PR[i])] = para_PR[i - 1];
        addr_shift[reinterpret_cast<long>(para_VBias[i])] = para_VBias[i - 1];
      }



             // parameter_blocks???????????????addr_shift??????????????????????????????????????????
      std::vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
      if (last_marginalization_info)
        delete last_marginalization_info;
      last_marginalization_info = marginalization_info; // ??????????????????????????????
      last_marginalization_parameter_blocks = parameter_blocks; // ?????????????????????????????????????????????????????????????????????????????????????????????


      break;
    }
    
    if(Frame_size != Lidar_WINDOW_SIZE) {
      for(int f=0; f<Frame_size; ++f){
        edgesPlan[f].clear();
        }   
    } 
 }
}

// LIO????????????
int lvi_fusion::service_LIO_update()
{
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "/world";
    /*** variables definition ***/
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > G, H_T_H, I_STATE;
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();

    cv::Mat matA1( 3, 3, CV_32F, cv::Scalar::all( 0 ) );
    cv::Mat matD1( 1, 3, CV_32F, cv::Scalar::all( 0 ) );
    cv::Mat matV1( 3, 3, CV_32F, cv::Scalar::all( 0 ) );
    cv::Mat matP( 6, 6, CV_32F, cv::Scalar::all( 0 ) );

    PointCloudXYZINormal::Ptr feats_undistort( new PointCloudXYZINormal() );  // ?????????????????????????????????
    PointCloudXYZINormal::Ptr feats_down( new PointCloudXYZINormal() );    // ????????????????????????
#if !use_livox
    PointCloudXYZINormal::Ptr feats_surf( new PointCloudXYZINormal() ); 
    PointCloudXYZINormal::Ptr feats_corn( new PointCloudXYZINormal() ); 
#endif
    PointCloudXYZINormal::Ptr laserCloudOri( new PointCloudXYZINormal() );  // ????????????????????????
    PointCloudXYZINormal::Ptr coeffSel( new PointCloudXYZINormal() ); // ?????????????????????

    /*** variables initialize ***/
    FOV_DEG = fov_deg + 10;
    HALF_FOV_COS = std::cos( ( fov_deg + 10.0 ) * 0.5 * PI_M / 180.0 );

    for ( int i = 0; i < laserCloudNum; i++ )
    {
        featsArray[ i ].reset( new PointCloudXYZINormal() );
    }

    std::shared_ptr< ImuProcess > p_imu( new ImuProcess() );
    m_imu_process = p_imu;
    //------------------------------------------------------------------------------------------------------
    ros::Rate rate( 5000 );
    bool      status = ros::ok();
    g_camera_lidar_queue.m_liar_frame_buf = &lidar_buffer;
    set_initial_state_cov( g_lio_state );
    while ( ros::ok() )
    {
        if ( flg_exit )
            break;
        ros::spinOnce();
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    
        while ( g_camera_lidar_queue.if_lidar_can_process() == false )
        {
           
            ros::spinOnce();
            std::this_thread::yield();
            std::this_thread::sleep_for( std::chrono::milliseconds( THREAD_SLEEP_TIM ) );
        }
         
        std::unique_lock< std::mutex > lock( m_mutex_lio_process );
        if ( 1 )
        {
            // printf_line;
            Common_tools::Timer tim;
       
            if ( sync_packages( Measures ) == 0 ) // ??????????????????????????????IMU??????
            {
                continue;
            }
            int lidar_can_update = 1;
            g_lidar_star_tim = frame_first_pt_time; 
            if ( flg_reset )
            {
                ROS_WARN( "reset when rosbag play back" );
                p_imu->Reset(); // ??????IMU?????????
                flg_reset = false;
                continue;
            }
           
            g_LiDAR_frame_index++;
            tim.tic( "Preprocess" );
            double t0, t1, t2, t3, t4, t5, match_start, match_time, solve_start, solve_time, pca_time, svd_time;
            match_time = 0;
            kdtree_search_time = 0;
            solve_time = 0;
            pca_time = 0;
            svd_time = 0;
            t0 = omp_get_wtime();
          
            // if(frame_count < WINDOW_SIZE)
            // {

            //     frame_count++;
                // int prev_frame = frame_count - 1;
                // Ps[frame_count] = Ps[prev_frame];
                // Vs[frame_count] = Vs[prev_frame];
                // Rs[frame_count] = Rs[prev_frame];
                // Bas[frame_count] = Bas[prev_frame];
                // Bgs[frame_count] = Bgs[prev_frame];
            // }
            p_imu->Process( Measures, g_lio_state, feats_undistort ); // ??????????????????IMU?????????
       
            // cout << "feats_undistort.size:"<< feats_undistort->points.size() <<endl;
            g_camera_lidar_queue.g_noise_cov_acc = p_imu->cov_acc;
            g_camera_lidar_queue.g_noise_cov_gyro = p_imu->cov_gyr;
            StatesGroup state_propagate( g_lio_state ); // state_propagate ??????????????????  g_lio_state IMU???????????????????????????
     
            // cout << "G_lio_state.last_update_time =  " << std::setprecision(10) << g_lio_state.last_update_time -g_lidar_star_tim  << endl;
            if ( feats_undistort->empty() || ( feats_undistort == NULL ) ) // ?????????????????????
            {
                frame_first_pt_time = Measures.lidar_beg_time;
                std::cout << "not ready for odometry" << std::endl;
                continue;
            }
  
            if ( ( Measures.lidar_beg_time - frame_first_pt_time ) < INIT_TIME )
            {
                flg_EKF_inited = false;
                std::cout << "||||||||||Initiallizing LiDAR||||||||||" << std::endl;
            }
            else
            {
                flg_EKF_inited = true;
            }
     

            /*** Compute the euler angle ***/
            Eigen::Vector3d euler_cur = RotMtoEuler( g_lio_state.rot_end );

            
#if 0
            // ???????????????????????????
            lasermap_fov_segment();//TODO ??????????????????????????????ikdtree????????????
#endif

#if use_livox
            // current scan ?????????
            downSizeFilterSurf.setInputCloud( feats_undistort );
            downSizeFilterSurf.filter( *feats_down );
#else 
            feats_down = feats_undistort;
             int feats = feats_down->points.size();
            feats_corn->clear();
            feats_surf->clear();
            for(int i=0;i<feats;i++){
                // cout << "feats_down->points[i].normal_x"<< feats_down->points[i].normal_x<<endl;
                if(feats_down->points[i].normal_x==1){
                    feats_corn->push_back(feats_down->points[i]);
                }
                else{
                   
                    feats_surf->push_back(feats_down->points[i]);
                }
            }
            // cout << "feats_down->points.size()"<< feats_down->points.size()<<endl;
            // cout << "feats_corn->points.size()"<< feats_corn->points.size()<<endl;
            // cout << "feats_surf->points.size()"<< feats_surf->points.size()<<endl;
            
#endif
            Frame lidar_frame;
#if use_livox
            lidar_frame.SurfaceCloud = feats_down;
#else
            lidar_frame.SurfaceCloud = feats_surf;
            lidar_frame.CornerCloud = feats_corn;
#endif
            lidar_frame.timeStamp =  Measures.lidar_end_time;
            lidar_frame.imuIntegrator = *p_imu;

            lidar_frame.P = g_lio_state.pos_end;
            lidar_frame.V = g_lio_state.vel_end;
            lidar_frame.Q = g_lio_state.rot_end;
            lidar_frame.ba = g_lio_state.bias_a;
            lidar_frame.bg = g_lio_state.bias_g;
            lidar_list->push_back(lidar_frame);
            vG = g_lio_state.gravity;


           
            // cout <<"Preprocess cost time: " << tim.toc("Preprocess") << endl;

#if use_livox
            /*** initialize the map kdtree ????????????????????????.????????????????????????***/ 
            if ( ( feats_down->points.size() > 1 ) && ( ikdtree.Root_Node == nullptr ) )
            {
                // std::vector<PointType> points_init = feats_down->points;
                ikdtree.set_downsample_param( filter_size_map_min );
                ikdtree.Build( feats_down->points );
                flg_map_initialized = true;
                continue;
            }
#else
            /*** initialize the map kdtree ????????????????????????.????????????????????????***/ 
            if ( ( feats_surf->points.size() > 1 ) && ( ikdtree.Root_Node == nullptr ) && ( feats_corn->points.size() > 1 ) && ( ikdtree_corn.Root_Node == nullptr ))
            {
                cout << "???????????????" << endl;
                // std::vector<PointType> points_init = feats_down->points;
                ikdtree.set_downsample_param( filter_size_map_min );
                ikdtree.Build( feats_surf->points );

                ikdtree_corn.set_downsample_param( filter_size_map_min/2.0 );
                ikdtree_corn.Build( feats_corn->points );
                flg_map_initialized = true;
                continue;
            }

#endif
            while(lidar_list->size()> Lidar_WINDOW_SIZE){
                lidar_list->pop_front();
            }
#if use_livox
            if ( ikdtree.Root_Node == nullptr ) // ???????????????????????????????????????
            {
                flg_map_initialized = false;
                std::cout << "~~~~~~~ Initialize Map iKD-Tree Failed! ~~~~~~~" << std::endl;
                continue;
            }
#else
            if ( ikdtree.Root_Node == nullptr ||  ikdtree_corn.Root_Node == nullptr) // ???????????????????????????????????????
            {
                flg_map_initialized = false;
                std::cout << "~~~~~~~ Initialize Map iKD-Tree Failed! ~~~~~~~" << std::endl;
                continue;
            }
#endif
            int featsFromMapNum = ikdtree.size(); // ???????????????
#if !use_livox
            int featsCornFromMapNum = ikdtree_corn.size(); // ???????????????
            cout << "???????????????" << featsCornFromMapNum << endl;
#endif

            int feats_down_size = feats_down->points.size();
#if !use_livox
            int feats_corn_size = feats_corn->points.size();
            int feats_surf_size = feats_surf->points.size();
#endif
            /*** ICP and iterated Kalman filter update ***/
#if use_livox
            PointCloudXYZINormal::Ptr coeffSel_tmpt( new PointCloudXYZINormal( *feats_down ) );  // ??????????????? abcd
            PointCloudXYZINormal::Ptr feats_down_updated( new PointCloudXYZINormal( *feats_down ) ); //??????????????????????????????
            std::vector< double >     res_last( feats_down_size, 1000.0 ); // initial
#else
            PointCloudXYZINormal::Ptr coeffSel_tmpt( new PointCloudXYZINormal( *feats_surf ) );  // ??????????????? abcd
            PointCloudXYZINormal::Ptr feats_down_updated( new PointCloudXYZINormal( *feats_surf ) ); //??????????????????????????????
            std::vector< double >     res_last( feats_surf_size, 1000.0 ); // initial

            PointCloudXYZINormal::Ptr coeffSel_tmpt_corn( new PointCloudXYZINormal( *feats_corn ) );  // ??????????????? abcd
            PointCloudXYZINormal::Ptr feats_down_updated_corn( new PointCloudXYZINormal( *feats_corn ) ); //??????????????????????????????
            std::vector< double >     res_last_corn( feats_corn_size, 1000.0 ); // initial
#endif
            if ( featsFromMapNum >= 5 )
            {
                t1 = omp_get_wtime();

                if ( m_if_publish_feature_map ) // ??????????????????
                {
                    PointVector().swap( ikdtree.PCL_Storage );// ??????
                    ikdtree.flatten( ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD );
                    featsFromMap->clear();
                    featsFromMap->points = ikdtree.PCL_Storage;
 #if !use_livox
                    PointVector().swap( ikdtree_corn.PCL_Storage );// ??????
                    ikdtree_corn.flatten( ikdtree_corn.Root_Node, ikdtree_corn.PCL_Storage, NOT_RECORD );
                
                    featsFromMap->points.insert(featsFromMap->points.end(), ikdtree_corn.PCL_Storage.begin(), ikdtree_corn.PCL_Storage.end());
 #endif
                    sensor_msgs::PointCloud2 laserCloudMap;
                    pcl::toROSMsg( *featsFromMap, laserCloudMap );
                    laserCloudMap.header.stamp = ros::Time::now(); // ros::Time().fromSec(last_timestamp_lidar);
                    // laserCloudMap.header.stamp.fromSec(Measures.lidar_end_time); // ros::Time().fromSec(last_timestamp_lidar);
                    laserCloudMap.header.frame_id = "world";
                    pubLaserCloudMap.publish( laserCloudMap );
                }
#if use_kf

    #if use_livox
                std::vector< bool >               point_selected_surf( feats_down_size, true );  // ??????????????????????????????????????????
                std::vector< std::vector< int > > pointSearchInd_surf( feats_down_size ); // ????????????
                std::vector< PointVector >        Nearest_Points( feats_down_size );  // ??????????????????????????????????????????????????????????????????????????????????????????kdtree???????????????

    #else
                std::vector< bool >               point_selected_surf( feats_surf_size, true );  // ??????????????????????????????????????????
                std::vector< bool >               point_selected_corn( feats_corn_size, true );  // ??????????????????????????????????????????
                std::vector< std::vector< int > > pointSearchInd_surf( feats_down_size ); // ????????????
                std::vector< PointVector >        Nearest_Points( feats_surf_size );  // ??????????????????????????????????????????????????????????????????????????????????????????kdtree???????????????
                std::vector< PointVector >        Nearest_Points_corn( feats_corn_size );  // ??????????????????????????????????????????????????????????????????????????????????????????kdtree???????????????
    #endif
                int  rematch_num = 0; // ??????????????????????????????
                bool rematch_en = 0; // ?????????????????????????????????
                flg_EKF_converged = 0;
                deltaR = 0.0;
                deltaT = 0.0;
                t2 = omp_get_wtime();
                double maximum_pt_range = 0.0; // ?????????????????????????????????
                // cout <<"Preprocess 2 cost time: " << tim.toc("Preprocess") << endl;
                for ( iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++ )
                {
                    tim.tic( "Iter" );
                    match_start = omp_get_wtime();
                    laserCloudOri->clear();
                    coeffSel->clear();

                    /** closest surface search and residual computation **/
                    //  ???????????????
#if use_livox
                    for ( int i = 0; i < feats_down_size; i += m_lio_update_point_step )
                    {
                        double     search_start = omp_get_wtime();
                        PointType &pointOri_tmpt = feats_down->points[ i ];
                        double     ori_pt_dis = sqrt( pointOri_tmpt.x * pointOri_tmpt.x + pointOri_tmpt.y * pointOri_tmpt.y + pointOri_tmpt.z * pointOri_tmpt.z ); // ?????????????????????
                        maximum_pt_range = std::max( ori_pt_dis, maximum_pt_range );
                        PointType &pointSel_tmpt = feats_down_updated->points[ i ];  // ???????????????

                        /* transform to world frame */
                        pointBodyToWorld( &pointOri_tmpt, &pointSel_tmpt );
                        std::vector< float > pointSearchSqDis_surf; // ??????????????????

                        auto &points_near = Nearest_Points[ i ];

                        //??????????????? || ???????????????????????????
                        if ( iterCount == 0 || rematch_en ) 
                        {
                            point_selected_surf[ i ] = true;
                            /** Find the closest surfaces in the map **/
                            ikdtree.Nearest_Search( pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf );
                            float max_distance = pointSearchSqDis_surf[ NUM_MATCH_POINTS - 1 ];
                            //  max_distance to add residuals
                            // ANCHOR - Long range pt stragetry
                            if ( max_distance > m_maximum_pt_kdtree_dis ) // ???????????????????????????????????????????????????????????????
                            {
                                point_selected_surf[ i ] = false;
                            }
                        }

                        kdtree_search_time += omp_get_wtime() - search_start;
                        if ( point_selected_surf[ i ] == false )
                            continue;

                        // match_time += omp_get_wtime() - match_start;
                        double pca_start = omp_get_wtime();
                        /// PCA (using minimum square method)
    #if 0
                        Eigen::Matrix<float, NUM_MATCH_POINTS, 3> A;
                        Eigen::Matrix<float, NUM_MATCH_POINTS, 1> b;
                        A.setZero();
                        b.setOnes();
                        b *= -1.0f;
                        for (int j = 0; j < NUM_MATCH_POINTS; j++)
                        {
                            A(j,0) = points_near[j].x;
                            A(j,1) = points_near[j].y;
                            A(j,2) = points_near[j].z;
                        }

                        Eigen::Matrix<float, 3, 1> normvec = A.colPivHouseholderQr().solve(b);
                        float n = normvec.norm();
                        float pa = normvec(0) / n;
                        float pb = normvec(1) / n;
                        float pc = normvec(2) / n;
                        float pd = 1.0 / n;

    #else
                        cv::Mat matA0( NUM_MATCH_POINTS, 3, CV_32F, cv::Scalar::all( 0 ) );
                        cv::Mat matB0( NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all( -1 ) );
                        cv::Mat matX0( NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all( 0 ) );

                        for ( int j = 0; j < NUM_MATCH_POINTS; j++ )
                        {
                            matA0.at< float >( j, 0 ) = points_near[ j ].x;
                            matA0.at< float >( j, 1 ) = points_near[ j ].y;
                            matA0.at< float >( j, 2 ) = points_near[ j ].z;
                        }

                        cv::solve( matA0, matB0, matX0, cv::DECOMP_QR ); // TODO

                        float pa = matX0.at< float >( 0, 0 );
                        float pb = matX0.at< float >( 1, 0 );
                        float pc = matX0.at< float >( 2, 0 );
                        float pd = 1;

                        float ps = sqrt( pa * pa + pb * pb + pc * pc );
                        pa /= ps;
                        pb /= ps;
                        pc /= ps;
                        pd /= ps;
    #endif
                        bool planeValid = true;
                        for ( int j = 0; j < NUM_MATCH_POINTS; j++ )
                        {
                            // ANCHOR -  Planar check
                            if ( fabs( pa * points_near[ j ].x + pb * points_near[ j ].y + pc * points_near[ j ].z + pd ) >
                                 m_planar_check_dis ) // Raw 0.05
                            {
                                // ANCHOR - Far distance pt processing// ????????????????????????????????????
                                // if ( ori_pt_dis < maximum_pt_range * 0.90 || ( ori_pt_dis < m_long_rang_pt_dis ) ) 
                                if(1)
                                {
                                    planeValid = false;
                                    point_selected_surf[ i ] = false;
                                    break;
                                }
                            }
                        }

                        if ( planeValid )
                        {
                            float pd2 = pa * pointSel_tmpt.x + pb * pointSel_tmpt.y + pc * pointSel_tmpt.z + pd;
                            // unuse
                            float s = 1 - 0.9 * fabs( pd2 ) /sqrt(ori_pt_dis);
                            // ANCHOR -  Point to plane distance
                            double acc_distance = ( ori_pt_dis < m_long_rang_pt_dis ) ? m_maximum_res_dis : 1.0;
                            // if ( pd2 < acc_distance ) 
                            if ( s>0.9)
                            {
                                // if(std::abs(pd2) > 5 * res_mean_last)
                                // {
                                //     point_selected_surf[i] = false;
                                //     res_last[i] = 0.0;
                                //     continue;
                                // }
                                point_selected_surf[ i ] = true;
                                coeffSel_tmpt->points[ i ].x = pa;
                                coeffSel_tmpt->points[ i ].y = pb;
                                coeffSel_tmpt->points[ i ].z = pc;
                                coeffSel_tmpt->points[ i ].intensity = pd2; // ???????????????????????????
                                res_last[ i ] = std::fabs( pd2 );
                            }
                            else
                            {
                                point_selected_surf[ i ] = false;
                            }
                        }
                        pca_time += omp_get_wtime() - pca_start;
                    }
#else
                    if(feats_surf_size< 100) continue;
                    for ( int i = 0; i < feats_surf_size; i += m_lio_update_point_step )
                    {
                        double     search_start = omp_get_wtime();
                        PointType &pointOri_tmpt = feats_surf->points[ i ];
                        double     ori_pt_dis = sqrt( pointOri_tmpt.x * pointOri_tmpt.x + pointOri_tmpt.y * pointOri_tmpt.y + pointOri_tmpt.z * pointOri_tmpt.z ); // ?????????????????????
                        maximum_pt_range = std::max( ori_pt_dis, maximum_pt_range );
                        PointType &pointSel_tmpt = feats_down_updated->points[ i ];

                        /* transform to world frame */
                        pointBodyToWorld( &pointOri_tmpt, &pointSel_tmpt );
                        std::vector< float > pointSearchSqDis_surf; // ??????????????????

                        auto &points_near = Nearest_Points[ i ];

                        //??????????????? || ???????????????????????????
                        if ( iterCount == 0 || rematch_en ) 
                        {
                            point_selected_surf[ i ] = true;
                            /** Find the closest surfaces in the map **/
                            ikdtree.Nearest_Search( pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf );
                            float max_distance = pointSearchSqDis_surf[ NUM_MATCH_POINTS - 1 ];
                            //  max_distance to add residuals
                            // ANCHOR - Long range pt stragetry
                            if ( max_distance > m_maximum_pt_kdtree_dis ) // ???????????????????????????????????????????????????????????????
                            {
                                point_selected_surf[ i ] = false;
                            }
                        }

                        kdtree_search_time += omp_get_wtime() - search_start;
                        if ( point_selected_surf[ i ] == false )
                            continue;

                        // match_time += omp_get_wtime() - match_start;
                        double pca_start = omp_get_wtime();
                        /// PCA (using minimum square method)
                        cv::Mat matA0( NUM_MATCH_POINTS, 3, CV_32F, cv::Scalar::all( 0 ) );
                        cv::Mat matB0( NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all( -1 ) );
                        cv::Mat matX0( NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all( 0 ) );

                        for ( int j = 0; j < NUM_MATCH_POINTS; j++ )
                        {
                            matA0.at< float >( j, 0 ) = points_near[ j ].x;
                            matA0.at< float >( j, 1 ) = points_near[ j ].y;
                            matA0.at< float >( j, 2 ) = points_near[ j ].z;
                        }

                        cv::solve( matA0, matB0, matX0, cv::DECOMP_QR ); // TODO

                        float pa = matX0.at< float >( 0, 0 );
                        float pb = matX0.at< float >( 1, 0 );
                        float pc = matX0.at< float >( 2, 0 );
                        float pd = 1;

                        float ps = sqrt( pa * pa + pb * pb + pc * pc );
                        pa /= ps;
                        pb /= ps;
                        pc /= ps;
                        pd /= ps;

                        bool planeValid = true;
                        for ( int j = 0; j < NUM_MATCH_POINTS; j++ )
                        {
                            // ANCHOR -  Planar check
                            if ( fabs( pa * points_near[ j ].x + pb * points_near[ j ].y + pc * points_near[ j ].z + pd ) >
                                 m_planar_check_dis ) // Raw 0.05
                            {
                                // ANCHOR - Far distance pt processing// ????????????????????????????????????
                                // if ( ori_pt_dis < maximum_pt_range * 0.90 || ( ori_pt_dis < m_long_rang_pt_dis ) ) 
                                if(1)
                                {
                                    planeValid = false;
                                    point_selected_surf[ i ] = false;
                                    break;
                                }
                            }
                        }

                        if ( planeValid )
                        {
                            float pd2 = pa * pointSel_tmpt.x + pb * pointSel_tmpt.y + pc * pointSel_tmpt.z + pd;
                            // unuse
                            // float s = 1 - 0.9 * fabs( pd2 ) /
                            //                   sqrt( sqrt( pointSel_tmpt.x * pointSel_tmpt.x + pointSel_tmpt.y * pointSel_tmpt.y +
                            //                               pointSel_tmpt.z * pointSel_tmpt.z ) );

                            float s = 1 - 0.9 * fabs( pd2 ) /sqrt(ori_pt_dis);
                            // ANCHOR -  Point to plane distance
                            double acc_distance = ( ori_pt_dis < m_long_rang_pt_dis ) ? m_maximum_res_dis : 1.0;
                            if ( pd2 < acc_distance ) 
                            // if ( s>0.9)
                            {
                                // if(std::abs(pd2) > 5 * res_mean_last)
                                // {
                                //     point_selected_surf[i] = false;
                                //     res_last[i] = 0.0;
                                //     continue;
                                // }
                                point_selected_surf[ i ] = true;
                                coeffSel_tmpt->points[ i ].x = pa;
                                coeffSel_tmpt->points[ i ].y = pb;
                                coeffSel_tmpt->points[ i ].z = pc;
                                coeffSel_tmpt->points[ i ].intensity = pd2; // ???????????????????????????
                                res_last[ i ] = std::fabs( pd2 );
                            }
                            else
                            {
                                point_selected_surf[ i ] = false;
                            }
                        }
                        pca_time += omp_get_wtime() - pca_start;
                    }

                    if(feats_corn_size< 10) continue;
                    for ( int i = 0; i < feats_corn_size; i += m_lio_update_point_step )
                    {
                        double     search_start = omp_get_wtime();
                        PointType &pointOri_tmpt = feats_corn->points[ i ];
                        double     ori_pt_dis = sqrt( pointOri_tmpt.x * pointOri_tmpt.x + pointOri_tmpt.y * pointOri_tmpt.y + pointOri_tmpt.z * pointOri_tmpt.z ); // ?????????????????????
                        maximum_pt_range = std::max( ori_pt_dis, maximum_pt_range );
                        PointType &pointSel_tmpt = feats_down_updated_corn->points[ i ];

                        /* transform to world frame */
                        pointBodyToWorld( &pointOri_tmpt, &pointSel_tmpt );
                        std::vector< float > pointSearchSqDis_surf; // ??????????????????

                        auto &points_near = Nearest_Points_corn[ i ];

                        //??????????????? || ???????????????????????????
                        if ( iterCount == 0 || rematch_en ) 
                        {
                            point_selected_corn[ i ] = true;
                            /** Find the closest surfaces in the map **/
                            ikdtree_corn.Nearest_Search( pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf );
                            float max_distance = pointSearchSqDis_surf[ NUM_MATCH_POINTS - 1 ];
                            //  max_distance to add residuals
                            // ANCHOR - Long range pt stragetry
                            if ( max_distance > m_maximum_pt_kdtree_dis ) // ???????????????????????????????????????????????????????????????
                            {
                                point_selected_corn[ i ] = false;
                            }
                        }

                        kdtree_search_time += omp_get_wtime() - search_start;
                        if ( point_selected_corn[ i ] == false )
                            continue;

                        // match_time += omp_get_wtime() - match_start;
                        double pca_start = omp_get_wtime();
                        /// PCA (using minimum square method)
                       Eigen::Matrix< double, 3, 3 > _matA1;
                       _matA1.setZero();
                          /*????????????*/
      
                        float cx = 0.;
                        float cy = 0.;
                        float cz = 0.;
                        for (int j = 0; j < 5; j++) {
                            cx += points_near[j].x;
                            cy += points_near[j].y;
                            cz += points_near[j].z;
                        }
                        cx /= 5.0;
                        cy /= 5.0;
                        cz /= 5.0;
                            /*????????????*/
                        float a11 = 0.;
                        float a12 = 0.;
                        float a13 = 0.;
                        float a22 = 0.;
                        float a23 = 0.;
                        float a33 = 0.;
                        for (int j = 0; j < 5; j++) {
                            float ax = points_near[j].x - cx;
                            float ay = points_near[j].y - cy;
                            float az = points_near[j].z - cz;

                            a11 += ax * ax;
                            a12 += ax * ay;
                            a13 += ax * az;
                            a22 += ay * ay;
                            a23 += ay * az;
                            a33 += az * az;
                        }
                        // TODO ??????????????????????????????
                        a11 /= 5.;
                        a12 /= 5.;
                        a13 /= 5.;
                        a22 /= 5.;
                        a23 /= 5.;
                        a33 /= 5.;

                        _matA1(0, 0) = a11;      _matA1(0, 1) = a12;      _matA1(0, 2) = a13;
                        _matA1(1, 0) = a12;      _matA1(1, 1) = a22;      _matA1(1, 2) = a23;
                        _matA1(2, 0) = a13;      _matA1(2, 1) = a23;      _matA1(2, 2) = a33;
                        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1); // ??????????????? ????????????????????????
                        Eigen::Vector3d unit_direction = saes.eigenvectors().col(2); // ????????????
                            // ?????????????????????????????????????????????????????????????????????????????????????????????????????????
                            // ????????????????????????????????????????????????????????????????????????s???????????????????????????
                            // ???????????????????????????????????????????????????????????????laserCloudOri???
                        if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) { 

                            float x0 = pointSel_tmpt.x;
                            float y0 = pointSel_tmpt.y;
                            float z0 = pointSel_tmpt.z;
                            // ??????????????????
                            float x1 = cx + 0.1 * unit_direction[0];
                            float y1 = cy + 0.1 * unit_direction[1];
                            float z1 = cz + 0.1 * unit_direction[2];
                            float x2 = cx - 0.1 * unit_direction[0];
                            float y2 = cy - 0.1 * unit_direction[1];
                            float z2 = cz - 0.1 * unit_direction[2];

                            // ???????????????[(x0-x1),(y0-y1),(z0-z1)]???[(x0-x2),(y0-y2),(z0-z2)]??????????????????????????????
                            // ??????????????????0.2*V1[0]??????[x0,y0,z0]?????????????????????????????????
                            /*
                                |  i       j      k   |
                            axb= | x0-x1  y0-y1  z0-z1 | = [(y0-y1)*(z0-z2)-(y0-y2)*(z0 -z1)]i+[(x0-x1)*(z0-z2)-(x0-x2)*(z0-z1)]j+[(x0-x1)*(y0-y2)-(x0-x2)*(y0-y1)]k
                                | x0-x2  y0-y2  z0-z2 |
                            */
                            float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                            + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                            + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));
                            // l12????????????0.2*(||V1[0]||)
                            // ??????????????????????????????????????????
                            float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                            // ??????????????????????????????????????????????????????????????????????????????????????????
                            float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                            float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                            float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;
                            // ?????????pointSel????????????????????????????????????????????????
                            // ???????????????=?????????????????????/????????????
                            float ld2 = a012 / l12;
                            // ????????????????????????????????????ld2?????????0????????????????????????
                            // ???????????????s=1???
                            float s = 1 - 0.9 * fabs(ld2);

                        

                   

                            // if ( s >0.1 )
                            {
                           
                                s = 1;  // ????????????
                                point_selected_corn[ i ] = true;
                                coeffSel_tmpt_corn->points[ i ].x = s * la;
                                coeffSel_tmpt_corn->points[ i ].y = s * lb;
                                coeffSel_tmpt_corn->points[ i ].z = s * lc;
                                coeffSel_tmpt_corn->points[ i ].intensity =  s * ld2; // ???????????????????????????
                                res_last_corn[ i ] = std::abs(  s * ld2 );
                            }
                            // else
                            // {
                            //     point_selected_corn[ i ] = false;
                            // }
                        }
                        pca_time += omp_get_wtime() - pca_start;
                    }
#endif 
                    tim.tic( "Stack" );
                    double total_residual = 0.0;
                    laserCloudSelNum = 0;
#if use_livox
                    // ???????????????????????????????????????
                    for ( int i = 0; i < coeffSel_tmpt->points.size(); i++ )
                    {
                           // ???????????? && ????????????????????????2    (????????????????????????????????????,????????????????????????????????????1.0)
                        if ( point_selected_surf[ i ] && ( res_last[ i ] <= 2.0 ) )
                        {
                            laserCloudOri->push_back( feats_down->points[ i ] );
                            coeffSel->push_back( coeffSel_tmpt->points[ i ] );
                            total_residual += res_last[ i ];
                            laserCloudSelNum++;
                        }
                    }
#else
                    for ( int i = 0; i < coeffSel_tmpt->points.size(); i++ )
                    {
                        if ( point_selected_surf[ i ] && ( res_last[ i ] <= 2.0 ) )
                        {
                            laserCloudOri->push_back( feats_surf->points[ i ] );
                            coeffSel->push_back( coeffSel_tmpt->points[ i ] );
                            total_residual += res_last[ i ];
                            laserCloudSelNum++;
                        }
                    }
                    for ( int i = 0; i < coeffSel_tmpt_corn->points.size(); i++ )
                    {
                        if ( point_selected_corn[ i ] && ( res_last_corn[ i ] <= 2.0 ) )
                        {
                            laserCloudOri->push_back( feats_corn->points[ i ] );
                            coeffSel->push_back( coeffSel_tmpt_corn->points[ i ] );
                            total_residual += res_last_corn[ i ];
                            laserCloudSelNum++;
                        }
                    }
             
#endif
        
                    res_mean_last = total_residual / laserCloudSelNum; // unuse

                    match_time += omp_get_wtime() - match_start;
                    solve_start = omp_get_wtime();

                    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/

                    Eigen::MatrixXd Hsub( laserCloudSelNum, 6 );  //????????????h????????????H  fast-lio ??????14
                    Eigen::VectorXd meas_vec( laserCloudSelNum ); //????????????h
                    Hsub.setZero();
                    
                    //??????????????????????????????
                    for ( int i = 0; i < laserCloudSelNum; i++ )
                    {
                        const PointType &laser_p = laserCloudOri->points[ i ];
                        Eigen::Vector3d  point_this( laser_p.x, laser_p.y, laser_p.z );
                        // TODO ???????????????????????????
                        point_this += Lidar_offset_to_IMU;  // ?????????imu-frame
                        Eigen::Matrix3d point_crossmat;
                        point_crossmat << SKEW_SYM_MATRIX( point_this );

                        /*** get the normal vector of closest surface/corner ***/
                        const PointType &norm_p = coeffSel->points[ i ];// ?????????
                        Eigen::Vector3d  norm_vec( norm_p.x, norm_p.y, norm_p.z ); // ?????????

                        /*** calculate the Measuremnt Jacobian matrix H ***/
                        // https://www.cnblogs.com/long5683/p/15401142.html
                        Eigen::Vector3d A( point_crossmat * g_lio_state.rot_end.transpose() * norm_vec );
                        Hsub.row( i ) << VEC_FROM_ARRAY( A ), norm_p.x, norm_p.y, norm_p.z;  // ????????????h????????????H  fast-lio ??????14

                        /*** Measuremnt: distance to the closest surface/corner ***/
                        meas_vec( i ) = -norm_p.intensity;  // -z
                    }

                    Eigen::Vector3d                           rot_add, t_add, v_add, bg_add, ba_add, g_add;//?????????:??????,??????,??????,???????????????????????????????????????????????? 
                    Eigen::Matrix< double, DIM_OF_STATES, 1 > solution; // ??????
                    Eigen::MatrixXd                           K( DIM_OF_STATES, laserCloudSelNum );// kalman??????
                    
                    /*** Iterative Kalman Filter Update ***/
                    if ( !flg_EKF_inited )
                    {
                        cout << ANSI_COLOR_RED_BOLD << "Run EKF init" << ANSI_COLOR_RESET << endl;
                        /*** only run in initialization period ***/
                        set_initial_state_cov( g_lio_state );
                    }
                    else
                    {
                        // cout << ANSI_COLOR_RED_BOLD << "Run EKF uph" << ANSI_COLOR_RESET << endl;
                        auto &&Hsub_T = Hsub.transpose();
                        H_T_H.block< 6, 6 >( 0, 0 ) = Hsub_T * Hsub;
                        //   fast-lio???????????????20?????????Kalman?????????????????????(??????R) : (H^T * R^-1 * H + P^-1)^-1 ?????????R???????????????????????????E??????
                        Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > &&K_1 =
                            ( H_T_H + ( g_lio_state.cov / LASER_POINT_COV ).inverse() ).inverse();

                        //??????????????????,?????????(20)Kalman??????(??????R)
                        K = K_1.block< DIM_OF_STATES, 6 >( 0, 0 ) * Hsub_T;
                        //?????????(18)????????????????????????????????????????????????
                        auto vec = state_propagate - g_lio_state;  //  x^k_k-x_k : Kalman??????????????????-??????(????????????)??????
                        solution = K * ( meas_vec - Hsub * vec.block< 6, 1 >( 0, 0 ) );  // Jk??????????????????????????????A(u)???u????????? A?????????E?????????
                        // double speed_delta = solution.block( 0, 6, 3, 1 ).norm();
                        // if(solution.block( 0, 6, 3, 1 ).norm() > 0.05 )
                        // {
                        //     solution.block( 0, 6, 3, 1 ) = solution.block( 0, 6, 3, 1 ) / speed_delta * 0.05;
                        // }

                        // ?????????18????????????,??????k+1???kalman??????????????????
                        g_lio_state = state_propagate + solution;
                        print_dash_board();
                        // cout << ANSI_COLOR_RED_BOLD << "Run EKF uph, vec = " << vec.head<9>().transpose() << ANSI_COLOR_RESET << endl;
                        rot_add = solution.block< 3, 1 >( 0, 0 ); // ????????????
                        t_add = solution.block< 3, 1 >( 3, 0 );  // ????????????
                        flg_EKF_converged = false;
                        // ??????????????????
                        if ( ( ( rot_add.norm() * 57.3 - deltaR ) < 0.01 ) && ( ( t_add.norm() * 100 - deltaT ) < 0.015 ) )
                        {
                            flg_EKF_converged = true;
                        }

                        deltaR = rot_add.norm() * 57.3;
                        deltaT = t_add.norm() * 100;
                    }
#else 

                    
                    // cout << "before"<< lidar_list->back().P.transpose()*1000<<endl;
                    ALoamEstimate(lidar_list);
                    // cout << "after"<< lidar_list->back().P.transpose()*1000<<endl;
                    g_lio_state.rot_end = lidar_list->back().Q;
                    g_lio_state.pos_end = lidar_list->back().P;
                    g_lio_state.vel_end = lidar_list->back().V;
                    g_lio_state.bias_g = lidar_list->back().bg;
                    g_lio_state.bias_a = lidar_list->back().ba;

#endif
                    // printf_line;
                    g_lio_state.last_update_time = Measures.lidar_end_time;
                    euler_cur = RotMtoEuler( g_lio_state.rot_end );// ????????????lidar??????????????????,???????????????????????????ros??????
                    dump_lio_state_to_log( m_lio_state_fp );
#if use_kf
                    /*** Rematch Judgement ???????????????????????????????????????***/
                    rematch_en = false;
                    if ( flg_EKF_converged || ( ( rematch_num == 0 ) && ( iterCount == ( NUM_MAX_ITERATIONS - 2 ) ) ) )
                    {
                        rematch_en = true;
                        rematch_num++;
                    }

                    /*** Convergence Judgements and Covariance Update ***/
                    //  ?????????????????????????????? : ??????FAST-LIO??????(19)
                    // if (rematch_num >= 10 || (iterCount == NUM_MAX_ITERATIONS - 1))
                    if ( rematch_num >= 2 || ( iterCount == NUM_MAX_ITERATIONS - 1 ) ) // Fast lio ori version.
                    {
                        if ( flg_EKF_inited )
                        {
                            /*** Covariance Update ***/ 
                            G.block< DIM_OF_STATES, 6 >( 0, 0 ) = K * Hsub; // ????????????(19)??? : K * H ????????????????????????????????????????????????????????????
                            g_lio_state.cov = ( I_STATE - G ) * g_lio_state.cov; //??????(19): (I-K*H)*P
                            // ???????????????
                            total_distance += ( g_lio_state.pos_end - position_last ).norm();
                            position_last = g_lio_state.pos_end;

                            // std::cout << "position: " << g_lio_state.pos_end.transpose() << " total distance: " << total_distance << std::endl;
                        }
                        solve_time += omp_get_wtime() - solve_start;
                        break;
                    }
                    solve_time += omp_get_wtime() - solve_start;

                    // cout << "Match cost time: " << match_time * 1000.0
                    //      << ", search cost time: " << kdtree_search_time*1000.0
                    //      << ", PCA cost time: " << pca_time*1000.0
                    //      << ", solver_cost: " << solve_time * 1000.0 << endl;
                    // cout <<"Iter cost time: " << tim.toc("Iter") << endl;
                }
#endif
                t3 = omp_get_wtime();

                /*** add new frame points to map ikdtree ***/
                PointVector points_history;  // ???ikd??????????????????????????????points_history???
                ikdtree.acquire_removed_points( points_history );
#if !use_livox
                ikdtree_corn.acquire_removed_points( points_history );
#endif
#if 1
                // ????????????????????????
                memset( cube_updated, 0, sizeof( cube_updated ) );
         
                // ????????????????????????
                for ( int i = 0; i < points_history.size(); i++ )
                {
                    PointType &pointSel = points_history[ i ];

                    // ????????????????????????????????????
                    int cubeI = int( ( pointSel.x + 0.5 * cube_len ) / cube_len ) + laserCloudCenWidth;
                    int cubeJ = int( ( pointSel.y + 0.5 * cube_len ) / cube_len ) + laserCloudCenHeight;
                    int cubeK = int( ( pointSel.z + 0.5 * cube_len ) / cube_len ) + laserCloudCenDepth;

                    if ( pointSel.x + 0.5 * cube_len < 0 )
                        cubeI--;
                    if ( pointSel.y + 0.5 * cube_len < 0 )
                        cubeJ--;
                    if ( pointSel.z + 0.5 * cube_len < 0 )
                        cubeK--;

                    // ?????????1?????????
                    if ( cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 && cubeJ < laserCloudHeight && cubeK >= 0 && cubeK < laserCloudDepth )
                    {
                        int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                        featsArray[ cubeInd ]->push_back( pointSel ); // ????????????????????????
                    }
                }
#endif
#if use_kf
    #if use_livox
                for ( int i = 0; i < feats_down_size; i++ )
                {
                    /* transform to world frame */
                    pointBodyToWorld( &( feats_down->points[ i ] ), &( feats_down_updated->points[ i ] ) );
                }
                t4 = omp_get_wtime();
                // ??????????????????????????????

                ikdtree.Add_Points( feats_down_updated->points, true );
    #else
                for ( int i = 0; i < feats_surf_size; i++ )
                {
                    /* transform to world frame */
                    pointBodyToWorld( &( feats_surf->points[ i ] ), &( feats_down_updated->points[ i ] ) );
                }

                for ( int i = 0; i < feats_corn_size; i++ )
                {
                    /* transform to world frame */
                    pointBodyToWorld( &( feats_corn->points[ i ] ), &( feats_down_updated_corn->points[ i ] ) );
                }
                t4 = omp_get_wtime();

                ikdtree.Add_Points( feats_down_updated->points, true );
                ikdtree_corn.Add_Points( feats_down_updated_corn->points, true );
    #endif
#else
                for ( int i = 0; i < lidar_list->front().SurfaceCloud->points.size(); i++ )
                {
                    /* transform to world frame */
                    // pointBodyToWorld( &( lidar_list->front().SurfaceCloud->points[ i ] ), &( feats_down_updated->points[ i ] ) );
                    Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
                    Eigen::Matrix3d exRbl =  Eigen::Matrix3d::Identity();
                    auto exPbl = Lidar_offset_to_IMU;
                    transformTobeMapped.topLeftCorner(3,3) = lidar_list->front().Q * exRbl;// TODO ?????????????????????R=1
                    transformTobeMapped.topRightCorner(3,1) = lidar_list->front().Q * exPbl + lidar_list->front().P;
                     pointAssociateToMap(&(lidar_list->front().SurfaceCloud->points[ i ] ), &( feats_down_updated->points[ i ] ), transformTobeMapped);
                }
                t4 = omp_get_wtime();
                // ??????????????????????????????
                ikdtree.Add_Points( feats_down_updated->points, true );
#endif

                kdtree_incremental_time = omp_get_wtime() - t4 + readd_time + readd_box_time + delete_box_time;
                t5 = omp_get_wtime();
            }

            /******* Publish current frame points in world coordinates:  *******/
            laserCloudFullRes2->clear();
            *laserCloudFullRes2 = dense_map_en ? ( *feats_undistort ) : ( *feats_down ); // ????????????????????????????????????

            int laserCloudFullResNum = laserCloudFullRes2->points.size();
  
            pcl::PointXYZI temp_point;
            laserCloudFullResColor->clear();
            {
                // ??????????????????????????????????????????
                for ( int i = 0; i < laserCloudFullResNum; i++ )
                {
                    RGBpointBodyToWorld( &laserCloudFullRes2->points[ i ], &temp_point );
                    laserCloudFullResColor->push_back( temp_point );
                }
                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg( *laserCloudFullResColor, laserCloudFullRes3 );
                // laserCloudFullRes3.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
                laserCloudFullRes3.header.stamp.fromSec( Measures.lidar_end_time );
                laserCloudFullRes3.header.frame_id = "world"; // world; camera_init
                pubLaserCloudFullRes.publish( laserCloudFullRes3 );
            }

            if ( 1 ) // append point cloud to global map. ????????????????????????????????? 
            {
                static std::vector< double > stastic_cost_time;
                Common_tools::Timer          tim;
                // tim.tic();
                // ANCHOR - RGB maps update
                wait_render_thread_finish(); // ????????????????????????
                if ( m_if_record_mvs ) // ?????????1???????????????
                {
                    std::vector< std::shared_ptr< RGB_pts > > pts_last_hitted;  //??????scan?????????grid????????????????????????
                    pts_last_hitted.reserve( 1e6 );
                    // ?????????scan???????????????global??????
                    m_number_of_new_visited_voxel = m_map_rgb_pts.append_points_to_global_map(
                        *laserCloudFullResColor, Measures.lidar_end_time - g_camera_lidar_queue.m_first_imu_time, &pts_last_hitted,
                        m_append_global_map_point_step );
                    m_map_rgb_pts.m_mutex_pts_last_visited->lock();
                    m_map_rgb_pts.m_pts_last_hitted = pts_last_hitted;
                    m_map_rgb_pts.m_mutex_pts_last_visited->unlock();
                }
                else
                {
                    m_number_of_new_visited_voxel = m_map_rgb_pts.append_points_to_global_map(
                        *laserCloudFullResColor, Measures.lidar_end_time - g_camera_lidar_queue.m_first_imu_time, nullptr,
                        m_append_global_map_point_step );
                }
                stastic_cost_time.push_back( tim.toc( " ", 0 ) );
            }


            // ?????????????????????????????????????????????????????????
            if(0) // Uncomment this code scope to enable the publish of effective points.
            {
                /******* Publish effective points *******/
                laserCloudFullResColor->clear();
                pcl::PointXYZI temp_point;
                for ( int i = 0; i < laserCloudSelNum; i++ )
                {
                    RGBpointBodyToWorld( &laserCloudOri->points[ i ], &temp_point );
                    laserCloudFullResColor->push_back( temp_point );
                }
                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg( *laserCloudFullResColor, laserCloudFullRes3 );
                // laserCloudFullRes3.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
                laserCloudFullRes3.header.stamp.fromSec( Measures.lidar_end_time ); //.fromSec(last_timestamp_lidar);
                laserCloudFullRes3.header.frame_id = "world";
                pubLaserCloudEffect.publish( laserCloudFullRes3 );
            }

            // ?????????????????????
            if(1){
                    pcl::PointCloud<pcl::PointXYZI>::Ptr tmpThisCloud(new pcl::PointCloud<pcl::PointXYZI>());
                    int tmp_count = laserCloudOri->points.size();
                    tmpThisCloud->resize(tmp_count);
                    for(int i=0; i<tmp_count; ++i ){
                        tmpThisCloud->points[i].x = laserCloudOri->points[i].x;
                        tmpThisCloud->points[i].y = laserCloudOri->points[i].y;
                        tmpThisCloud->points[i].z = laserCloudOri->points[i].z;
                        tmpThisCloud->points[i].intensity= laserCloudOri->points[i].intensity;
                    }


                sensor_msgs::PointCloud2 laserCloud;
                pcl::toROSMsg( *tmpThisCloud, laserCloud );
                laserCloud.header.stamp.fromSec( Measures.lidar_end_time ); //.fromSec(last_timestamp_lidar);
                laserCloud.header.frame_id = "base_link";
                pubLaserCloud.publish( laserCloud );
            }


            /******* Publish Maps:  *******/
            sensor_msgs::PointCloud2 laserCloudMap;
            pcl::toROSMsg( *featsFromMap, laserCloudMap );
            laserCloudMap.header.stamp.fromSec( Measures.lidar_end_time ); // ros::Time().fromSec(last_timestamp_lidar);
            laserCloudMap.header.frame_id = "world";
            pubLaserCloudMap.publish( laserCloudMap );

            /******* Publish Odometry ******/
            geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw( euler_cur( 0 ), euler_cur( 1 ), euler_cur( 2 ) );
            odomAftMapped.header.frame_id = "world";
            odomAftMapped.child_frame_id = "/base_link";
            odomAftMapped.header.stamp = ros::Time().fromSec( Measures.lidar_end_time ); // ros::Time().fromSec(last_timestamp_lidar); ros::Time::now();
            odomAftMapped.pose.pose.orientation.x = geoQuat.x;
            odomAftMapped.pose.pose.orientation.y = geoQuat.y;
            odomAftMapped.pose.pose.orientation.z = geoQuat.z;
            odomAftMapped.pose.pose.orientation.w = geoQuat.w;
            odomAftMapped.pose.pose.position.x = g_lio_state.pos_end( 0 );
            odomAftMapped.pose.pose.position.y = g_lio_state.pos_end( 1 );
            odomAftMapped.pose.pose.position.z = g_lio_state.pos_end( 2 );

            pubOdomAftMapped.publish( odomAftMapped );

            static tf::TransformBroadcaster br;
            tf::Transform                   transform;
            tf::Quaternion                  q;
            transform.setOrigin(
                tf::Vector3( odomAftMapped.pose.pose.position.x, odomAftMapped.pose.pose.position.y, odomAftMapped.pose.pose.position.z ) );
            q.setW( odomAftMapped.pose.pose.orientation.w );
            q.setX( odomAftMapped.pose.pose.orientation.x );
            q.setY( odomAftMapped.pose.pose.orientation.y );
            q.setZ( odomAftMapped.pose.pose.orientation.z );
            transform.setRotation( q );
            br.sendTransform( tf::StampedTransform( transform, ros::Time().fromSec( Measures.lidar_end_time ), "world", "/base_link" ) );

            msg_body_pose.header.stamp = ros::Time::now();
     
            msg_body_pose.pose.position.x = g_lio_state.pos_end( 0 );
            msg_body_pose.pose.position.y = g_lio_state.pos_end( 1 );
            msg_body_pose.pose.position.z = g_lio_state.pos_end( 2 );
            msg_body_pose.pose.orientation.x = geoQuat.x;
            msg_body_pose.pose.orientation.y = geoQuat.y;
            msg_body_pose.pose.orientation.z = geoQuat.z;
            msg_body_pose.pose.orientation.w = geoQuat.w;

            /******* Publish Path ********/
            msg_body_pose.header.frame_id = "world";
            if ( frame_num > 10 )
            {
                path.poses.push_back( msg_body_pose );
            }
            pubPath.publish( path );
            
            /*** save debug variables ***/
            frame_num++;
            aver_time_consu = aver_time_consu * ( frame_num - 1 ) / frame_num + ( t5 - t0 ) / frame_num;
            // aver_time_consu = aver_time_consu * 0.8 + (t5 - t0) * 0.2;
            T1[ time_log_counter ] = Measures.lidar_beg_time;
            s_plot[ time_log_counter ] = aver_time_consu;
            s_plot2[ time_log_counter ] = kdtree_incremental_time;
            s_plot3[ time_log_counter ] = kdtree_search_time;
            s_plot4[ time_log_counter ] = fov_check_time;
            s_plot5[ time_log_counter ] = t5 - t0;
            s_plot6[ time_log_counter ] = readd_box_time;
            time_log_counter++;
            fprintf( m_lio_costtime_fp, "%.5f %.5f\r\n", g_lio_state.last_update_time - g_camera_lidar_queue.m_first_imu_time, t5 - t0 );
            fflush( m_lio_costtime_fp );
        }
        status = ros::ok();
        rate.sleep();
    }
    return 0;
}
