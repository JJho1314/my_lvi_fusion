// Author:   Qian Chenglong
#ifndef LIDAR_FACTOR_H
#define LIDAR_FACTOR_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include "utility.h"

const double aaa = 1.0;

/** \brief Ceres Cost Funtion between PointCloud Flat Feature and Map Cloud
 */
struct Cost_NavState_IMU_Plan_Vec
{
    Cost_NavState_IMU_Plan_Vec(Eigen::Vector3d  _p, 
							   Eigen::Vector3d  _p_proj,
							   const Eigen::Matrix4d& Tbl,
							   Eigen::Matrix<double, 3, 3> _sqrt_information):
                               point(std::move(_p)),
							   point_proj(std::move(_p_proj)), 
							   sqrt_information(std::move(_sqrt_information)){
      Eigen::Matrix3d m3d = Tbl.topLeftCorner(3,3);
      qbl = Eigen::Quaterniond(m3d).normalized();
      qlb = qbl.conjugate();
      Pbl = Tbl.topRightCorner(3,1);
      Plb = -(qlb * Pbl);
    }

    template <typename T>
    bool operator()(const T *PRi, T *residual) const {
      Eigen::Matrix<T, 3, 1> cp{T(point.x()), T(point.y()), T(point.z())};
	  Eigen::Matrix<T, 3, 1> cp_proj{T(point_proj.x()), T(point_proj.y()), T(point_proj.z())};

     

	  	Eigen::Quaternion<T> q_wb{PRi[6], PRi[3], PRi[4], PRi[5]};
		Eigen::Matrix<T, 3, 1> t_wb{PRi[0], PRi[1], PRi[2]};
      Eigen::Quaternion<T> q_wl = q_wb * qbl.cast<T>();
      Eigen::Matrix<T, 3, 1> t_wl = q_wb * Pbl.cast<T>() + t_wb;
      Eigen::Matrix<T, 3, 1> P_to_Map = q_wl * cp + t_wl;

	  Eigen::Map<Eigen::Matrix<T, 3, 1> > eResiduals(residual);
      eResiduals = P_to_Map - cp_proj;

	//   cout << "lidar 残差：" << P_to_Map - cp_proj<<endl;
	//   T _weight = T(1) - T(0.9) * (P_to_Map - cp_proj).norm() /ceres::sqrt(
    //           ceres::sqrt( P_to_Map(0) * P_to_Map(0) +
    //                        P_to_Map(1) * P_to_Map(1) +
    //                        P_to_Map(2) * P_to_Map(2) ));
	//   eResiduals *= _weight;
	//   eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

      return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d& curr_point_,
                                       const Eigen::Vector3d&  p_proj_,
                                       const Eigen::Matrix4d& Tbl,
									   const Eigen::Matrix<double, 3, 3> sqrt_information_) {
      return (new ceres::AutoDiffCostFunction<Cost_NavState_IMU_Plan_Vec, 3, 7>(
              new Cost_NavState_IMU_Plan_Vec(curr_point_, p_proj_, Tbl, sqrt_information_)));
    }

    Eigen::Vector3d point;
	Eigen::Vector3d point_proj;
    Eigen::Quaterniond qbl, qlb;
    Eigen::Vector3d Pbl, Plb;
	Eigen::Matrix<double, 3, 3> sqrt_information;
};

struct LidarPlaneNormFactor 
{

	LidarPlaneNormFactor(Eigen::Vector3d curr_point_, // 当前点的局部坐标
						Eigen::Vector3d plane_unit_norm_, // 平面单位法向量 pa pb pc
						 double negative_OA_dot_norm_,
						 Eigen::Vector3d tbl_, //  IMU lidar
						 Eigen::Matrix3d Rbl_)  // pd
						 : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
							negative_OA_dot_norm(negative_OA_dot_norm_),tbl(tbl_), Rbl(Rbl_){}

	template <typename T>
	bool operator()(const T *PQ , T *residual) const{
		// Twl
		Eigen::Quaternion<T> q_w_curr{PQ[6], PQ[3], PQ[4], PQ[5]};
		Eigen::Matrix<T, 3, 1> t_w_curr{PQ[0], PQ[1], PQ[2]};

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		cp = Rbl* cp + tbl;                // Pbi = Tbl * Pli = Rbl  * Pli + tbl
		point_w = q_w_curr * cp + t_w_curr; // 转到世界坐标系

		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));

		// float s = 1 - 0.9 * fabs( pd2 ) /sqrt(point_w.norm() );
		// 计算点到平面的距离，这里去掉了原来的权重s
		residual[0] = (norm.dot(point_w) + T(negative_OA_dot_norm))*aaa; // pd2
		//  cout << "lidar 残差：" << residual[0]<<endl;
		return true;

	}


	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
									   const double negative_OA_dot_norm_,Eigen::Vector3d tbl_, //  IMU lidar
						 Eigen::Matrix3d Rbl_)
	{
		return (new ceres::AutoDiffCostFunction<LidarPlaneNormFactor, 1, 7>(
			new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_,tbl_,Rbl_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	Eigen::Vector3d tbl;  //  IMU lidar
	Eigen::Matrix3d Rbl;
	double negative_OA_dot_norm;
};



#endif