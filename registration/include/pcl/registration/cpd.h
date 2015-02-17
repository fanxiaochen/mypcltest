/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *  Copyright (C) 2008 Ben Gurion University of the Negev, Beer Sheva, Israel.
 *
 *  All rights reserved
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met
 *
 *   * The use for research only (no for any commercial application).
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
 */

#ifndef PCL_REGISTRATION_CPD_H_
#define PCL_REGISTRATION_CPD_H_

#include <pcl/common/common.h>
#include <pcl/registration/registration.h>
//#include <pcl/registration/matching_candidate.h>

namespace pcl
{
  ///** \brief Compute the mean point density of a given point cloud.
  //  * \param[in] cloud pointer to the input point cloud
  //  * \param[in] max_dist maximum distance of a point to be considered as a neighbor
  //  * \param[in] nr_threads number of threads to use (default = 1, only used if OpenMP flag is set)
  //  * \return the mean point density of a given point cloud
  //  */
  //template <typename PointT> inline float
  //getMeanPointDensity (const typename pcl::PointCloud<PointT>::ConstPtr &cloud, float max_dist, int nr_threads = 1);

  ///** \brief Compute the mean point density of a given point cloud.
  //  * \param[in] cloud pointer to the input point cloud
  //  * \param[in] indices the vector of point indices to use from \a cloud
  //  * \param[in] max_dist maximum distance of a point to be considered as a neighbor
  //  * \param[in] nr_threads number of threads to use (default = 1, only used if OpenMP flag is set)
  //  * \return the mean point density of a given point cloud
  //  */
  //template <typename PointT> inline float
  //getMeanPointDensity (const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const std::vector <int> &indices,
  //  float max_dist, int nr_threads = 1);
  
  
  namespace registration
  {
    /** \brief FPCSInitialAlignment computes corresponding four point congruent sets as described in:
    * "4-points congruent sets for robust pairwise surface registration", Dror Aiger, Niloy Mitra, Daniel Cohen-Or.
    * ACM Transactions on Graphics, vol. 27(3), 2008
    * \author P.W.Theiler
    * \ingroup registration
    */
    template <typename PointT>
    class CPD
    {
    public:
      typedef boost::shared_ptr<CPD<PointT> > Ptr;
      typedef boost::shared_ptr<const CPD<PointT> > ConstPtr;

      typedef pcl::PointCloud<PointT> PointCloud;
      typedef typename PointCloud::Ptr PointCloudPtr;
      typedef typename PointCloud::ConstPtr PointCloudConstPtr;

      typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VectorXf;
      typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> Matrix3f;
      typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf;

      typedef enum {RIGID, NONRIGID} RegType;

      typedef struct  
      {
        MatrixXf r_;
        VectorXf t_;
        float s_;
        float sigma2_;
      }RigidParas;

      typedef struct  
      {
        MatrixXf w_;
        float sigma2_;
        float lambda_;
        float beta_;
      }NonRigidParas;

    public:
      /** \brief Constructor.
        * Resets the maximum number of iterations to 0 thus forcing an internal computation if not set by the user.
        * Sets the number of RANSAC iterations to 1000 and the standard transformation estimation to TransformationEstimation3Point.
        */
      CPD ();

      /** \brief Destructor. */
      virtual ~CPD ()
      {};

      inline void
      setSourceCloud (const PointCloudPtr& source_cloud)
      {
        source_cloud_ = source_cloud;
      }

      inline void
      setTargetCloud (const PointCloudPtr& target_cloud)
      {
        target_cloud_ = target_cloud;
      }

      inline PointCloudConstPtr
      getSourceCloud () const 
      {
        return (source_cloud_);
      }
      
      inline PointCloudConstPtr
      getTargetCloud () const
      {
        return (target_cloud_);
      }
      

      inline void
      setMaxIteration (size_t max_iters)
      {
        max_iters_ = max_iters;
      }

      inline size_t
      getMaxInteration () const
      {
        return (max_iters_);
      }

      inline void
      setVarianceTol (float var_tol)
      {
        var_tol_ = var_tol;
      }

      inline float
      getVarianceTol () const
      {
        return (var_tol_);
      }

      inline void
      setEnergyTol (float energy_tol)
      {
        energy_tol_ = energy_tol;
      }

      inline float
      getEnergyTol () const
      {
        return (energy_tol_);
      }

      inline void
      setOutlierWeight (float w)
      {
        w_ = w;
      }

      inline float
      getOutlierWeight () const
      {
        return (w_);
      }

      inline void
      setLambda (float lambda)
      {
        nrigid_paras_.lambda_ = lambda;
      }

      inline float
      getLambda () const
      {
        return (nrigid_paras_.lambda_);
      }

      inline void
      setBeta (float beta)
      {
        nrigid_paras_.beta_ = beta;
      }

      inline float
      getBeta () const
      {
        return (nrigid_paras_.beta_);
      }


      inline void
      setRegType (RegType reg_type)
      {
        reg_type_ = reg_type;
      }

      inline RegType
      getRegType () const
      {
        return (reg_type_);
      }

      void
      run ();

      void
      computeRigid ();

      void
      computeNonRigid ();


    protected:

      inline void
      initTransform ()
      {
        transform_ = source_mat_;
      }

      inline void
      updateSourceMat ()
      {
        source_mat_ = transform_;
      }

      void 
      initMats ();

      void
      updateClouds ();

      void
      normalize ();

      void
      denormalize ();


    protected:
      void
      initialize ();

      void
      e_step ();

      void
      m_step ();

      void
      align ();

      void
      computeCorres ();

      void
      constructG ();

      float
      computeGaussianExp (size_t m, size_t n);

      float
      energy ();


    protected:

      PointCloudPtr source_cloud_;
      PointCloudPtr target_cloud_;

      Matrix3f source_mat_;
      Matrix3f target_mat_;

      Matrix3f transform_;

      MatrixXf corres_;
      RegType reg_type_;

      size_t max_iters_;
      float energy_tol_;
      float var_tol_;
      float w_;

      VectorXf source_means_;
      float source_scale_;

      VectorXf target_means_;
      float target_scale_;

      RigidParas rigid_paras_;
      NonRigidParas nrigid_paras_;

      MatrixXf p1_;
      MatrixXf pt1_;
      MatrixXf px_;

      MatrixXf g_;

    };
  }; // namespace registration  
}; // namespace pcl 

#include <pcl/registration/impl/cpd.hpp>

#endif // PCL_REGISTRATION_CPD_H_
