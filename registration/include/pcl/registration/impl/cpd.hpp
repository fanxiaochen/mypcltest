/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id: cpd.hpp 5663 2015-02-17 13:49:39Z Xiaochen Fan $
 *
 */

#ifndef PCL_REGISTRATION_IMPL_CPD_HPP_
#define PCL_REGISTRATION_IMPL_CPD_HPP_

#include <pcl/point_types.h>
#include <pcl/registration/cpd.h>
//#include "cpd.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT>
pcl::registration::CPD<PointT>::CPD ()
  : max_iters_(50),
  var_tol_(1e-3),
  energy_tol_(1e-3),
  w_(0),
  reg_type_(RIGID)
{
  nrigid_paras_.lambda_ = 2.0;
  nrigid_paras_.beta_ = 2.0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
pcl::registration::CPD<PointT>::initMats ()
{
  size_t source_size = source_cloud_->size ();
  size_t target_size = target_cloud_->size ();

  source_mat_.resize (source_size, 3);
  target_mat_.resize (target_size, 3);

  for (size_t i = 0, i_end = source_cloud_->size (); i < i_end; ++ i)
  {
    const PointT& point = source_cloud_->at (i);
    source_mat_.row (i) << point.x, point.y, point.z;
  }

  for (size_t i = 0, i_end = target_cloud_->size (); i < i_end; ++ i)
  {
    const PointT& point = target_cloud_->at (i);
    target_mat_.row (i) << point.x, point.y, point.z;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
pcl::registration::CPD<PointT>::updateClouds ()
{
  for (size_t i = 0, i_end = source_cloud_->size (); i < i_end; ++ i)
  {
    PointT& point = source_cloud_->at (i);
    point.x = source_mat_.row(i)[0];
    point.y = source_mat_.row(i)[1];
    point.z = source_mat_.row(i)[2];
  }

  for (size_t i = 0, i_end = target_cloud_->size (); i < i_end; ++ i)
  {
    PointT& point = target_cloud_->at (i);
    point.x = target_mat_.row(i)[0];
    point.y = target_mat_.row(i)[1];
    point.z = target_mat_.row(i)[2];
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
pcl::registration::CPD<PointT>::normalize ()
{
  int M = source_mat_.rows();
  int N = target_mat_.rows();

  std::cout << "M:" << M << std::endl;
  std::cout << "N:" << N << std::endl;

  source_means_ = source_mat_.colwise().mean();
  target_means_ = target_mat_.colwise().mean();

  source_mat_ = source_mat_ - source_means_.transpose().replicate(M, 1);
  target_mat_ = target_mat_ - target_means_.transpose().replicate(N, 1);

  source_scale_ = sqrt(source_mat_.array().square().sum() / M);
  target_scale_ = sqrt(target_mat_.array().square().sum() / N);

  source_mat_ = source_mat_ / source_scale_;
  target_mat_ = target_mat_ / target_scale_;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
pcl::registration::CPD<PointT>::denormalize ()
{
  int M = source_mat_.rows();
  int N = target_mat_.rows();

  source_mat_ = source_mat_ * target_scale_ + target_means_.transpose().replicate(M, 1);
  target_mat_ = target_mat_ * target_scale_ + target_means_.transpose().replicate(N, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
pcl::registration::CPD<PointT>::computeCorres ()
{
  int M = source_mat_.rows();
  int N = target_mat_.rows();

  corres_.setZero(M, N);

  float sigma2 = 0;
  if (reg_type_ == RIGID)
    sigma2 = rigid_paras_.sigma2_;
  else
    sigma2 = nrigid_paras_.sigma2_;

  for (size_t n = 0; n < N; n ++)
  {
    std::vector<float> t_exp;
    float sum_exp = 0;
    float c = pow((2*M_PI*sigma2), 0.5*3) * (w_/(1-w_)) * (float(M)/N);
    for (size_t m = 0; m < M; m ++)
    {
      float m_exp = computeGaussianExp(m, n);
      t_exp.push_back(m_exp);
      sum_exp += m_exp;
    }

    for (size_t m = 0; m < M; m ++)
    {
      corres_(m, n) = t_exp.at(m) / (sum_exp + c);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> float
pcl::registration::CPD<PointT>::computeGaussianExp (size_t m, size_t n)
{
  float sigma2 = 0;
  if (reg_type_ == RIGID)
    sigma2 = rigid_paras_.sigma2_;
  else
    sigma2 = nrigid_paras_.sigma2_;

  VectorXf vec = VectorXf(transform_.row(m) - target_mat_.row(n));
  float g_exp = exp(-vec.squaredNorm()/(2*sigma2));
  return g_exp;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> float
pcl::registration::CPD<PointT>::energy ()
{
  int M = source_mat_.rows();
  int N = target_mat_.rows();

  float sigma2 = 0;
  if (reg_type_ == RIGID)
    sigma2 = rigid_paras_.sigma2_;
  else
    sigma2 = nrigid_paras_.sigma2_;

  float e = 0;

  for (size_t n = 0; n < N; n ++)
  {
    float sp = 0;
    for (size_t m = 0; m < M; m ++)
    {
      sp += computeGaussianExp(m, n);
    }

    sp += pow((2*M_PI*sigma2), 0.5*3) * (w_/(1-w_)) * (float(M)/N);

    e += -log(sp);

  }

  e += N * 3 * log(sigma2) / 2;

  return e;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
pcl::registration::CPD<PointT>::align ()
{
  int M = source_mat_.rows();

  if (reg_type_ == RIGID)
  {
    transform_ = (rigid_paras_.s_) * (source_mat_) * (rigid_paras_.r_).transpose() + 
      VectorXf(M).setOnes() * (rigid_paras_.t_).transpose();
  }
  else
  {
    transform_ = source_mat_ + g_ * nrigid_paras_.w_;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
pcl::registration::CPD<PointT>::e_step ()
{
  int M = source_mat_.rows();
  int N = target_mat_.rows();

  computeCorres();
  p1_ = corres_ * VectorXf(N).setOnes();
  pt1_ = corres_.transpose() * VectorXf(M).setOnes();
  px_ = corres_ * target_mat_;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
pcl::registration::CPD<PointT>::m_step ()
{
  int M = source_mat_.rows();
  int N = target_mat_.rows();

  if (reg_type_ == RIGID)
  {
    float n_p = p1_.sum();

    VectorXf mu_x = target_mat_.transpose() * pt1_ / n_p;
    VectorXf mu_y = source_mat_.transpose() * p1_ / n_p;

    Matrix3f X_hat = target_mat_ - MatrixXf(VectorXf(N).setOnes() * mu_x.transpose());
    Matrix3f Y_hat = source_mat_ - MatrixXf(VectorXf(M).setOnes() * mu_y.transpose());

    MatrixXf A = (px_-p1_*mu_x.transpose()).transpose() * 
      (source_mat_ - MatrixXf(VectorXf(M).setOnes() * mu_y.transpose()));

    Eigen::JacobiSVD<MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXf U = svd.matrixU();
    MatrixXf V = svd.matrixV();

    float det_uv = MatrixXf(U*V.transpose()).determinant();
    MatrixXf C = MatrixXf::Identity(3, 3);
    C(2, 2) = det_uv;
    rigid_paras_.r_ = U * C * V.transpose();

    float s_upper = MatrixXf(A.transpose()*rigid_paras_.r_).trace();
    float s_lower = MatrixXf(Y_hat.transpose()*p1_.asDiagonal()*Y_hat).trace();
    rigid_paras_.s_ =  s_upper / s_lower; 

    rigid_paras_.t_ = mu_x - rigid_paras_.s_ * rigid_paras_.r_ * mu_y;

    float tr_f = MatrixXf(X_hat.transpose()*pt1_.asDiagonal()*X_hat).trace();
    float tr_b = MatrixXf(A.transpose()*rigid_paras_.r_).trace();
    rigid_paras_.sigma2_ = (tr_f - rigid_paras_.s_ * tr_b) / (n_p * 3);
    rigid_paras_.sigma2_ = abs(rigid_paras_.sigma2_);
  }
  else
  {
    float n_p = p1_.sum();

    MatrixXf A = (p1_.asDiagonal()*g_ + nrigid_paras_.lambda_*nrigid_paras_.sigma2_*MatrixXf::Identity(M, M));
    MatrixXf B = px_ - p1_.asDiagonal() * source_mat_;
    nrigid_paras_.w_ = A.inverse() * B;

    align();

    nrigid_paras_.sigma2_ = 1/(n_p*3) * ((target_mat_.transpose()*pt1_.asDiagonal()*target_mat_).trace() -
      2*(px_.transpose()*transform_).trace() + (transform_.transpose()*p1_.asDiagonal()*transform_).trace());
    nrigid_paras_.sigma2_ = abs(nrigid_paras_.sigma2_);

  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
pcl::registration::CPD<PointT>::initialize ()
{
  int M = source_mat_.rows();
  int N = target_mat_.rows();

  if (reg_type_ == RIGID)
  {
    // initialization
    rigid_paras_.r_ = MatrixXf::Identity(3, 3);
    rigid_paras_.t_ = VectorXf::Zero(3, 1);
    rigid_paras_.s_ = 1;

    float sigma_sum = M*(target_mat_.transpose()*target_mat_).trace() + 
      N*(source_mat_.transpose()*source_mat_).trace() - 
      2*(target_mat_.colwise().sum())*(source_mat_.colwise().sum()).transpose();
    rigid_paras_.sigma2_ = sigma_sum / (3*N*M);

    initTransform();
  }
  else
  {
    nrigid_paras_.w_ = MatrixXf::Zero(M, 3);

    float sigma_sum = M*(target_mat_.transpose()*target_mat_).trace() + 
      N*(source_mat_.transpose()*source_mat_).trace() - 
      2*(target_mat_.colwise().sum())*(source_mat_.colwise().sum()).transpose();
    nrigid_paras_.sigma2_ = sigma_sum / (3*N*M);

    initTransform();
    constructG();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
pcl::registration::CPD<PointT>::constructG ()
{
  int M = source_mat_.rows();

  g_ = MatrixXf::Zero(M, M);

  for (size_t i = 0; i < M; i ++)
  {
    for (size_t j = 0; j < M; j ++)
    {
      g_(i, j) = exp(-VectorXf(source_mat_.row(i)-source_mat_.row(j)).squaredNorm()/(2*nrigid_paras_.beta_*nrigid_paras_.beta_));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
pcl::registration::CPD<PointT>::run ()
{
  initMats();

  if (reg_type_ == RIGID)
    computeRigid();
  else
    computeNonRigid();

  updateClouds();

  return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
pcl::registration::CPD<PointT>::computeRigid ()
{
  reg_type_ = RIGID;

  size_t iter_num = 0;
  float e_tol = 10 + energy_tol_;
  float e = 0;

  normalize();
  initialize();

  while (iter_num < max_iters_ && e_tol > energy_tol_ && rigid_paras_.sigma2_ > 10 * var_tol_)
  {
    e_step();

    float old_e = e;
    e = energy();
    e_tol = abs((e - old_e) / e);

    m_step();
    align();

    iter_num ++;	
  }

  computeCorres();
  updateSourceMat();
  denormalize();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
pcl::registration::CPD<PointT>::computeNonRigid ()
{
  reg_type_ = NONRIGID;

  size_t iter_num = 0;
  float e_tol = 10 + energy_tol_;
  float e = 0;

  normalize();
  initialize();

  while (iter_num < max_iters_ && e_tol > energy_tol_ && nrigid_paras_.sigma2_ > 10 * var_tol_)
  {

    e_step();

    float old_e = e;
    e = energy();
    e += nrigid_paras_.lambda_/2 * (nrigid_paras_.w_.transpose()*g_*nrigid_paras_.w_).trace();
    e_tol = abs((e - old_e) / e);

    m_step();

    iter_num ++;	
  }

  computeCorres();
  updateSourceMat();
  denormalize();
}

#define PCL_INSTANTIATE_CPD(T) template class PCL_EXPORTS pcl::registration::CPD<T>;

#endif  // PCL_REGISTRATION_IMPL_CPD_HPP_

