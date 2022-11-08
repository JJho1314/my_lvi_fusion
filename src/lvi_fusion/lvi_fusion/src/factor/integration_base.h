/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include "loam/IMU_Processing.hpp"
#include "utility.h"
#include <ceres/ceres.h>

// TODO 
const Eigen::Vector3d vG{0.0, 0.0, 9.8};  //g在世界坐标系下的值


class IntegrationBase
{
  public:Utility::
    IntegrationBase() = delete;
    IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
        : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
            jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, covariance{Eigen::Matrix<double, 15, 15>::Zero()},
          sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()}, delta_q{Eigen::Quaterniond::Identity()}, delta_v{Eigen::Vector3d::Zero()}

    {
        // 测量噪声协方差，认为每时刻是固定的
        // 噪声包括高斯白噪声、bias随机游走噪声
        // TODO 这里需要调整
        noise = Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) =  (COV_ACC_NOISE_DIAG * COV_ACC_NOISE_DIAG) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) =  (COV_GYRO_NOISE_DIAG * COV_GYRO_NOISE_DIAG) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) =  (COV_ACC_NOISE_DIAG * COV_ACC_NOISE_DIAG) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) =  (COV_GYRO_NOISE_DIAG * COV_GYRO_NOISE_DIAG) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) =  (COV_BIAS_ACC_NOISE_DIAG * COV_BIAS_ACC_NOISE_DIAG) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) =  (COV_BIAS_GYRO_NOISE_DIAG * COV_BIAS_GYRO_NOISE_DIAG) * Eigen::Matrix3d::Identity();
    }

    /**
     * 添加一帧IMU，中值积分传播
    */
    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        /**
         * IMU中值积分传播
         * 1、前一时刻状态计算当前时刻状态，PVQ，其中Ba，Bg保持不变
         * 2、计算当前时刻的误差Jacobian，误差协方差 todo
        */
        propagate(dt, acc, gyr);
    }

    /**
     * IMU重传播，状态清零，设置新偏置，中值积分传播
    */
    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {

        // cout<<"sum_dt:"<<sum_dt<<", "<<typeid(sum_dt).name()<<endl;
        sum_dt = double(0.0);

        acc_0 = linearized_acc;

        gyr_0 = linearized_gyr;

        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();

        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;

        jacobian.setIdentity();
        covariance.setZero();

        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    /**
     * 中值积分
     * 1、前一时刻状态计算当前时刻状态，PVQ，其中Ba，Bg保持不变
     * 2、计算当前时刻的误差相对于预积分起始时刻的Jacobian，增量误差协方差 todo
    */
    void midPointIntegration(double _dt, 
                            const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        //ROS_INFO("midpoint integration");
        // 注:以下计算PVQ都是每时刻世界坐标系下(第一帧IMU系)的量，加速度、角速度都是IMU系下的量
        // 前一时刻加速度
        Eigen::Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
        // 前一时刻与当前时刻角速度中值
        Eigen::Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        // 当前时刻旋转位姿Q
        result_delta_q = delta_q * Eigen::Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        // 当前时刻加速度
        Eigen::Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
        // 前一时刻与当前时刻加速度中值
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        // 更新当前时刻P, V, 其中Ba, Bg保持不变
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
        result_delta_v = delta_v + un_acc * _dt;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;         

        if(update_jacobian)
        {
            Eigen::Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
            Eigen::Vector3d a_0_x = _acc_0 - linearized_ba;
            Eigen::Vector3d a_1_x = _acc_1 - linearized_ba;
            Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;

            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_0_x<<0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
            R_a_1_x<<0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;

            // 当前时刻误差对前一时刻误差的微分
            Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt + 
                                  -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            F.block<3, 3>(0, 6) = Eigen::MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 12) = -1.0 * Eigen::MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt + 
                                  -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
            F.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();
            //cout<<"A"<<endl<<A<<endl;

            // 当前误差对噪声的微分
            Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15,18);
            V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) =  0.5 * Eigen::MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(3, 9) =  0.5 * Eigen::MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = Eigen::MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(12, 15) = Eigen::MatrixXd::Identity(3,3) * _dt;

            //step_jacobian = F;
            //step_V = V;
            // 累乘，当前时刻误差对初始时刻误差的微分
            jacobian = F * jacobian;
            // 当前时刻误差协方差
            // 相邻时刻误差线性传递方程：误差k = Fk-1*误差k-1 + Vk-1*测量噪声k-1
            // 对应协方差关系：协方差k = Fk-1*协方差k-1*Fk-1转置 + Vk-1*测量噪声*Vk-1转置，认为测量噪声每时刻都是一样的，测量噪声包括高斯白噪声、bias随机游走噪声
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }

    }

    /**
     * IMU中值积分传播
     * 1、前一时刻状态计算当前时刻状态，PVQ，Ba，Bg
     * 2、计算当前时刻的误差Jacobian，误差协方差 todo
    */
    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        Eigen::Vector3d result_delta_p;
        Eigen::Quaterniond result_delta_q;
        Eigen::Vector3d result_delta_v;
        Eigen::Vector3d result_linearized_ba;
        Eigen::Vector3d result_linearized_bg;

        /**
         * 中值积分
         * 1、前一时刻状态计算当前时刻状态，PVQ，其中Ba，Bg保持不变
         * 2、计算当前时刻的误差Jacobian，误差协方差 todo
        */

        midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
        //                    linearized_ba, linearized_bg);
        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;  
     
    }

    /**
     * 预积分残差
     * 用预积分起止时刻对应的视觉里程计位姿（还包括速度、偏置）变换，与预积分量相减构建残差
    */
    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    {
        Eigen::Matrix<double, 15, 1> residuals;
        // 预积分结束时刻误差相对于预积分起始时刻误差的微分
        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        // 预积分起始时刻，加速度偏置的误差
        Eigen::Vector3d dba = Bai - linearized_ba;
        // 预积分起始时刻，角速度偏置的误差
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        // 预积分起止时间段内的旋转量Q，带噪声修正
        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        // 预积分起止时间段内的速度差量V，带噪声修正
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        // 预积分起止时间段内的平移量P，带噪声修正
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        // 用预积分起止时刻对应的视觉里程计位姿变换，与预积分量相减构建残差
        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * vG * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        // 旋转这里只存了四元数的虚部
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (vG * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }

    double dt;
    Eigen::Vector3d acc_0, gyr_0;
    Eigen::Vector3d acc_1, gyr_1;

    const Eigen::Vector3d linearized_acc, linearized_gyr;
    Eigen::Vector3d linearized_ba, linearized_bg;

    Eigen::Matrix<double, 15, 15> jacobian, covariance;
    Eigen::Matrix<double, 15, 15> step_jacobian;
    Eigen::Matrix<double, 15, 18> step_V;
    Eigen::Matrix<double, 18, 18> noise;

    double sum_dt;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;

    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;

};
