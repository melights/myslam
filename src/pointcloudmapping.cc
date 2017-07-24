/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */
#include <fstream>
#include "pointcloudmapping.h"
#include <MapPoint.h>
#include <KeyFrame.h>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utility.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/surface/poisson.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include "Converter.h"
//#include "StereoEfficientLargeScale.h"
using namespace cv;
//using namespace std;
#define STRIDE 4

///////////CUDA things/////////////
cv::Mat Q(4, 4, CV_32F) ;
enum RefImage {LeftRefImage, RightRefImage};
struct CostVolumeParams {

    uint8_t min_disp;
    uint8_t max_disp;
    uint8_t num_disp_layers;
    uint8_t method; // 0 for AD, 1 for ZNCC
    uint8_t win_r;
    RefImage ref_img;

};

struct PrimalDualParams {

    uint32_t num_itr;

    float alpha;
    float beta;
    float epsilon;
    float lambda;
    float aux_theta;
    float aux_theta_gamma;

    /* With preconditoining, we don't need these. */
    float sigma;
    float tau;
    float theta;

};

cv::Mat stereoCalcu(int _m, int _n, float* _left_img, float* _right_img, CostVolumeParams _cv_params, PrimalDualParams _pd_params);

////////////////////
bool simulation = true;
Eigen::Matrix3f intrinsics;
PointCloudMapping::PointCloudMapping(double resolution_)
{
    FileStorage fs("./Q.xml", FileStorage::READ);
    if (!fs.isOpened())
    {
        printf("Failed to open file ../Q.xml");
    }
    fs["Q"] >> Q;

    this->resolution = resolution_;
    voxel.setLeafSize(resolution, resolution, resolution);
    globalMap = boost::make_shared<PointCloudmono>();
    triangles_ptr = boost::make_shared<pcl::PolygonMesh>();
    //namedWindow("Disparity", WINDOW_NORMAL);
    //cloudViewerThread = make_shared<thread>(bind(&PointCloudMapping::Cloud_Viewer, this));
    surfaceViewerThread = make_shared<thread>(bind(&PointCloudMapping::Surface_Viewer, this));
    updateThread = make_shared<thread>(bind(&PointCloudMapping::update, this));
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    cloudViewerThread->join();
    surfaceViewerThread->join();
}

void PointCloudMapping::insertKeyFrame(KeyFrame *kf, cv::Mat &left_img, cv::Mat &right_img)
{

    cout << "receive a keyframe, id = " << kf->mnId << endl;
    unique_lock<mutex> lck(keyframeMutex);
    //keyframes.push_back( kf );
    keyframe = kf;
    left = left_img;
    right = right_img;
    // colorImgs.push_back( color.clone() );
    // depthImgs.push_back( depth.clone() );

    keyFrameUpdated.notify_one();
}

pcl::PointCloud<PointCloudMapping::PointT>::Ptr PointCloudMapping::generatePointCloud()
{
    Mat left_grey, right_gery;
    cvtColor(left, left_grey, cv::COLOR_RGB2GRAY);
    cvtColor(right, right_gery, cv::COLOR_RGB2GRAY);

    // Ptr<StereoBM> bm = StereoBM::create(16,9);
    // Rect roi1, roi2;
    // Mat disp,disp8U;
    //     bm->setROI1(roi1);
    // bm->setROI2(roi2);
    // bm->setPreFilterCap(1);
    // bm->setBlockSize(23);
    // bm->setMinDisparity(0);
    // bm->setNumDisparities(48);
    // bm->setTextureThreshold(0);
    // bm->setUniquenessRatio(0);
    // bm->setSpeckleWindowSize(0);
    // bm->setSpeckleRange(0);
    // bm->setDisp12MaxDiff(1);
    // bm->compute(left_grey, right_gery, disp);

    Ptr<StereoBM> bm = StereoBM::create(96, 55);
    Rect roi1, roi2;
    Mat disp, disp8U;
    //     bm->setROI1(roi1);
    // bm->setROI2(roi2);
    // bm->setPreFilterCap(1);
    // bm->setBlockSize(55);
    // bm->setMinDisparity(0);
    // bm->setNumDisparities(96);
    // bm->setTextureThreshold(0);
    // bm->setUniquenessRatio(0);
    // bm->setSpeckleWindowSize(0);
    // bm->setSpeckleRange(0);
    // bm->setDisp12MaxDiff(1);
    bm->compute(left_grey, right_gery, disp);

    // Mat disp,disp8U;
    // StereoEfficientLargeScale elas(0,128);
    // elas(left_grey,right_gery,disp,100);

    double minVal;
    double maxVal;
    minMaxLoc(disp, &minVal, &maxVal);
    disp.convertTo(disp8U, CV_8UC1, 255 / (maxVal - minVal));
    imshow("Disparity", disp8U);
    cv::waitKey();

    double px, py, pz;
    uchar pr, pg, pb;
    PointCloud::Ptr tmp(new PointCloud());
    for (int i = 0; i < left.rows - 3; i = i + 3)
    {
        const short *disp_ptr = disp.ptr<short>(i);
        for (int j = 0; j < left.cols - 3; j = j + 3)
        {
            double d = static_cast<double>(disp_ptr[j]) / 16;
            //double d = (double)disp.at<Vec3b>(i, j)[0];
            //std::cout<<d<<" ";
            //std::cout<<"i:"<<i<<"j:"<<j<<std::endl;
            if (d == -1 || d == 0 || d > 100 || d < 0)
                continue; //Discard bad pixels

            pz = keyframe->mbf / d;
            px = (static_cast<double>(j) - keyframe->cx) * pz / keyframe->fx;
            py = (static_cast<double>(i) - keyframe->cy) * pz / keyframe->fy;
            if (simulation)
            {
                if ((i - 240) * (i - 240) + (j - 320) * (j - 320) > 67600) //circle mask for simulation data
                    continue;
            }
            PointT point;
            point.x = px;
            point.y = py;
            point.z = pz;
            point.b = left.at<Vec3b>(i, j)[0];
            point.g = left.at<Vec3b>(i, j)[1];
            point.r = left.at<Vec3b>(i, j)[2];
            if (simulation)
            {
                if (point.b == 64 && point.g == 64 && point.r == 64) //remove grey background for simulation data
                    continue;
            }
            tmp->points.push_back(point);
        }
    }
    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(keyframe->GetPose());
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud(*tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;
    //cloud->width = (int) cloud->points.size();
    //cloud->height = 1;
    //pcl::io::savePCDFile( "/home/long/surface_reconstruction/PCL/data/testfilter.pcd", *cloud );

    cout << "generate point cloud for kf " << keyframe->mnId << ", size=" << cloud->points.size() << endl;
    PointCloud::Ptr filter1(new PointCloud());
    PointCloud::Ptr filter2(new PointCloud());

    pcl::RadiusOutlierRemoval<PointT> outrem;
    outrem.setInputCloud(cloud);
    outrem.setRadiusSearch(0.08);
    outrem.setMinNeighborsInRadius(5);
    // apply filter
    outrem.filter(*filter1);

    //   pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
    //   statistical_filter.setMeanK(80);
    //   statistical_filter.setStddevMulThresh(1.0);
    //   statistical_filter.setInputCloud(filter1);
    //   statistical_filter.filter(*filter2);

    return filter1;
}


pcl::PointCloud<PointCloudMapping::PointTmono>::Ptr PointCloudMapping::generatePointCloudCUDA()
{
    Mat left_grey, right_gery;
    cvtColor(left, left_grey, cv::COLOR_RGB2GRAY);
    cvtColor(right, right_gery, cv::COLOR_RGB2GRAY);
Mat left_32F, right_32F;
    left_grey.convertTo(left_32F, CV_32F);
    right_gery.convertTo(right_32F, CV_32F);

    CostVolumeParams cv_params;
    cv_params.min_disp = 0;
    cv_params.max_disp = 64;
    cv_params.method = 1;
    cv_params.win_r = 10;
    cv_params.ref_img = LeftRefImage;

    PrimalDualParams pd_params;
    pd_params.num_itr = 150; // 500
    pd_params.alpha = 0.1; // 10.0 0.01
    pd_params.beta = 1.0; // 1.0
    pd_params.epsilon = 0.1; // 0.1
    pd_params.lambda = 1e-2; // 1e-3
    pd_params.aux_theta = 10; // 10
    pd_params.aux_theta_gamma = 1e-6; // 1e-6
    std::cout<<left_32F.size()<<std::endl;
    cv::Mat result =  stereoCalcu(left_32F.rows, left_32F.cols, (float*)left_32F.data, (float*)right_32F.data, cv_params, pd_params);

    result.convertTo(result, CV_32F, cv_params.max_disp);
    cv::Mat image3D;
    cv::reprojectImageTo3D(result, image3D, Q);

    PointCloudmono::Ptr point_cloud_ptr(new PointCloudmono());
    for(int x = 0; x < left.cols - STRIDE; x += STRIDE)
    {
        for(int y = 0; y < left.rows - STRIDE; y += STRIDE)
        {
            if (simulation)
            {
                if ((y - 240) * (y - 240) + (x - 320) * (x - 320) > 67600) //circle mask for simulation data
                    continue;
                if (left.at<Vec3b>(y, x)[0] == 64 && left.at<Vec3b>(y, x)[1] == 64 && left.at<Vec3b>(y, x)[2] == 64) //remove grey background for simulation data
                    continue;
            }

            cv::Vec3f point3D = image3D.at<cv::Vec3f>(y,x);
            PointTmono basic_point;
            basic_point.x = point3D.val[0];
            basic_point.y = point3D.val[1];
            basic_point.z = point3D.val[2];
            //std::cout<<basic_point<<std::endl;
            if(cvIsInf(point3D.val[0]) || cvIsInf(point3D.val[1]) || cvIsInf(point3D.val[2]))
                ;//
            else
            {
 //               cout << point3D.val[0] << " " << point3D.val[1] << " " << point3D.val[2] << endl ;
                point_cloud_ptr->points.push_back(basic_point);

            }
        }
    }

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(keyframe->GetPose());
    PointCloudmono::Ptr cloud(new PointCloudmono);
    pcl::transformPointCloud(*point_cloud_ptr, *cloud, T.inverse().matrix());
    cloud->is_dense = false;
    //cloud->width = (int) cloud->points.size();
    //cloud->height = 1;
    //pcl::io::savePCDFile( "/home/long/surface_reconstruction/PCL/data/testfilter.pcd", *cloud );

    cout << "generate point cloud for kf " << keyframe->mnId << ", size=" << cloud->points.size() << endl;


    //   pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
    //   statistical_filter.setMeanK(80);
    //   statistical_filter.setStddevMulThresh(1.0);
    //   statistical_filter.setInputCloud(filter1);
    //   statistical_filter.filter(*filter2);

    return cloud;
}
pcl::PointCloud<PointCloudMapping::PointTmono>::Ptr PointCloudMapping::generatePointCloudmono()
{
    Mat left_grey, right_gery;
    cvtColor(left, left_grey, cv::COLOR_RGB2GRAY);
    cvtColor(right, right_gery, cv::COLOR_RGB2GRAY);

    Ptr<StereoBM> bm = StereoBM::create(96, 55);
    Rect roi1, roi2;
    Mat disp, disp8U;

    bm->compute(left_grey, right_gery, disp);

    // Mat disp,disp8U;
    // StereoEfficientLargeScale elas(0,128);
    // elas(left_grey,right_gery,disp,100);

    double minVal;
    double maxVal;
    minMaxLoc(disp, &minVal, &maxVal);
    disp.convertTo(disp8U, CV_8UC1, 255 / (maxVal - minVal));
    //imshow("Disparity", disp8U);
    //cv::waitKey(10);
    double px, py, pz;
    uchar pr, pg, pb;
    PointCloudmono::Ptr tmp(new PointCloudmono());
    for (int i = 0; i < left.rows - STRIDE; i += STRIDE)
    {
        const short *disp_ptr = disp.ptr<short>(i);
        for (int j = 0; j < left.cols - STRIDE; j += STRIDE)
        {
            double d = static_cast<double>(disp_ptr[j]) / 16;
            //double d = (double)disp.at<Vec3b>(i, j)[0];
            //std::cout<<d<<" ";
            //std::cout<<"i:"<<i<<"j:"<<j<<std::endl;
            if (d == -1 || d == 0 || d > 100 || d < 0)
                continue; //Discard bad pixels

            pz = keyframe->mbf / d;
            px = (static_cast<double>(j) - keyframe->cx) * pz / keyframe->fx;
            py = (static_cast<double>(i) - keyframe->cy) * pz / keyframe->fy;
            if (simulation)
            {
                if ((i - 240) * (i - 240) + (j - 320) * (j - 320) > 67600) //circle mask for simulation data
                    continue;
            }
            PointTmono point;
            point.x = px;
            point.y = py;
            point.z = pz;

            if (simulation)
            {
                if (left.at<Vec3b>(i, j)[0] == 64 && left.at<Vec3b>(i, j)[1] == 64 && left.at<Vec3b>(i, j)[2] == 64) //remove grey background for simulation data
                    continue;
            }
            tmp->points.push_back(point);
        }
    }
    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(keyframe->GetPose());
    PointCloudmono::Ptr cloud(new PointCloudmono);
    pcl::transformPointCloud(*tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;
    //cloud->width = (int) cloud->points.size();
    //cloud->height = 1;
    //pcl::io::savePCDFile( "/home/long/surface_reconstruction/PCL/data/testfilter.pcd", *cloud );

    cout << "generate point cloud for kf " << keyframe->mnId << ", size=" << cloud->points.size() << endl;


    //   pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
    //   statistical_filter.setMeanK(80);
    //   statistical_filter.setStddevMulThresh(1.0);
    //   statistical_filter.setInputCloud(filter1);
    //   statistical_filter.filter(*filter2);

    return cloud;
}

void PointCloudMapping::Cloud_Viewer()
{

    pcl::visualization::CloudViewer cloudviewer("3D Reconstruction");
    // cloudviewer.registerKeyboardCallback(&PointCloudMapping::keyboard_callback, *this);

    // surface_viewer.setBackgroundColor(0, 0, 0); //设置窗口颜色
    // //surface_viewer.setRepresentationToSurfaceForAllActors(); //网格模型以面片形式显示
    // surface_viewer.setRepresentationToPointsForAllActors(); //网格模型以点形式显示
    // //surface_viewer.setRepresentationToWireframeForAllActors();  //网格模型以线框图模式显示
    // surface_viewer.addCoordinateSystem(1); //设置坐标系,参数为坐标显示尺寸
    // surface_viewer.initCameraParameters();
    while (1)
    {
        int64 t = getTickCount();
        {
            unique_lock<mutex> lck_shutdown(shutDownMutex);
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated(keyFrameUpdateMutex);
            keyFrameUpdated.wait(lck_keyframeUpdated);
        }
        size_t N = 0;
        {
            unique_lock<mutex> lck(keyframeMutex);
            N = keyframes.size();
        }

        PointCloud::Ptr p = generatePointCloud();
        //*globalMap += *p;

        // PointCloud::Ptr tmp(new PointCloud());
        // voxel.setInputCloud(globalMap);
        // voxel.filter(*tmp);
        //globalMap->swap( *tmp );
        // cloudviewer.removeAllShapes();
        // cloudviewer.removeAllPointClouds();
        //         pcl::ModelCoefficients cylinder_coeff;
        // cylinder_coeff.values.resize (7);    // We need 7 values
        // cylinder_coeff.values[0] = keyframe->GetCameraCenter().at<float>(0);
        // cylinder_coeff.values[1] = keyframe->GetCameraCenter().at<float>(1);
        // cylinder_coeff.values[2] = keyframe->GetCameraCenter().at<float>(2);
        // Mat R,Rvec;
        // R=keyframe->GetRotation();
        // cv::Rodrigues(R, Rvec);
        // cylinder_coeff.values[3] = Rvec.at<float>(0)*10;
        // cylinder_coeff.values[4] = Rvec.at<float>(1)*10;
        // cylinder_coeff.values[5] = Rvec.at<float>(2)*10;
        // cylinder_coeff.values[6] = 2;

        //         cloudviewer.addCylinder(cylinder_coeff);
        //         cloudviewer.addPointCloud( globalMap );
        cout << "globalMap size=" << globalMap->points.size() << endl;

        cloudviewer.showCloud(globalMap);
        t = getTickCount() - t;
        printf("Time elapsed: %fms\n", t * 1000 / getTickFrequency());
        //cloudviewer.spinOnce(100);
    }
}

void PointCloudMapping::update()
{
    while (1)
    {
        int64 t = getTickCount();
        {
            unique_lock<mutex> lck_shutdown(shutDownMutex);
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated(keyFrameUpdateMutex);
            keyFrameUpdated.wait(lck_keyframeUpdated);
        }
        // size_t N = 0;
        // {
        //     unique_lock<mutex> lck(keyframeMutex);
        //     N = keyframes.size();
        // }

        PointCloudmono::Ptr p = generatePointCloudmono();
*globalMap += *p;
//*globalMap = *p;
        
        PointCloudmono::Ptr tmp(new PointCloudmono());
        voxel.setInputCloud(globalMap);
        voxel.filter(*tmp);
        globalMap->swap( *tmp );

        //* the data should be available in cloud

        // Normal estimation*
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(globalMap);
        n.setInputCloud(globalMap);
        n.setSearchMethod(tree);
        n.setKSearch(10);
        n.compute(*normals);
        //* normals should not contain the point normals + surface curvatures

        // Concatenate the XYZ and normal fields*
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
        pcl::concatenateFields(*globalMap, *normals, *cloud_with_normals);
        //* cloud_with_normals = cloud + normals

        // Create search tree*
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
        tree2->setInputCloud(cloud_with_normals);

        // Initialize objects
        pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;

        // Set the maximum distance between connected points (maximum edge length)
        gp3.setSearchRadius(10);

        // Set typical values for the parameters
        gp3.setMu(2.5);
        gp3.setMaximumNearestNeighbors(100);
        gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
        gp3.setMinimumAngle(M_PI / 18);       // 10 degrees
        gp3.setMaximumAngle(2 * M_PI / 3);    // 120 degrees
        gp3.setNormalConsistency(false);

pcl::PolygonMesh::Ptr triangles_tmp(new pcl::PolygonMesh());
        // Get result
        gp3.setInputCloud(cloud_with_normals);
        gp3.setSearchMethod(tree2);
        gp3.reconstruct(*triangles_tmp);
        unique_lock<mutex> lck(reconstructionMutex);
        triangles_ptr.swap(triangles_tmp);
        t = getTickCount() - t;
        printf("Total point: %d, Time elapsed: %fms\n", globalMap->size(), t * 1000 / getTickFrequency());
        //cloudviewer.spinOnce(100);
    }

}
void PointCloudMapping::Surface_Viewer()
{
    pcl::visualization::PCLVisualizer surface_viewer("Surface");
    surface_viewer.setBackgroundColor(0, 0, 0); //设置窗口颜色
    //surface_viewer.setRepresentationToSurfaceForAllActors(); //网格模型以面片形式显示
    //surface_viewer.setRepresentationToPointsForAllActors(); //网格模型以点形式显示
    surface_viewer.registerKeyboardCallback( &PointCloudMapping::keyboard_callback, *this );
    surface_viewer.setRepresentationToWireframeForAllActors();  //网格模型以线框图模式显示
    surface_viewer.setShowFPS(1);
    surface_viewer.addCoordinateSystem(1); //设置坐标系,参数为坐标显示尺寸
    surface_viewer.initCameraParameters();
    //surface_viewer.addPolygonMesh(*triangles_ptr,"my"); //设置所要显示的网格对象
    while (1)
    {
        //pcl::PolygonMesh triangles_cp=triangles;
        surface_viewer.removePolygonMesh("my"); //设置所要显示的网格对象
        surface_viewer.removeAllShapes();
        {
            unique_lock<mutex> lck(reconstructionMutex);
            if(triangles_ptr->polygons.size()>0)
                surface_viewer.addPolygonMesh(*triangles_ptr,"my"); //设置所要显示的网格对象
        }
                surface_viewer.setRepresentationToWireframeForAllActors();  //网格模型以线框图模式显示

        cv::Mat rVec, tVec;
        if(mCameraPose.rows > 0){
            {
            unique_lock<mutex> lock(mMutexCamera);
            cv::Mat Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
            cv::Mat twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
            cv::Rodrigues(Rwc, rVec);
            tVec = twc;
            }

            pcl::ModelCoefficients cylinder_coeff;
            //std::cout<<tVec<<std::endl;
            cylinder_coeff.values.resize (7);    // We need 7 values
            cylinder_coeff.values[0] = tVec.at<float>(0);
            cylinder_coeff.values[1] = tVec.at<float>(1);
            cylinder_coeff.values[2] = tVec.at<float>(2);
            // Mat R,Rvec;
            // R=keyframe->GetRotation();
            // cv::Rodrigues(R, Rvec);
            cylinder_coeff.values[3] = rVec.at<float>(0);
            cylinder_coeff.values[4] = rVec.at<float>(1);
            cylinder_coeff.values[5] = rVec.at<float>(2);
            cylinder_coeff.values[6] = 1;
            cylinder_coeff.values[7] = 1;
            cylinder_coeff.values[8] = 1;
            cylinder_coeff.values[9] = 0.3;

                 surface_viewer.addCube(cylinder_coeff);
        }

        // surface_viewer.addPolygonMesh(triangles,to_string(t));
        surface_viewer.spinOnce();
        //cloudviewer.spinOnce(100);
    }
}

void PointCloudMapping::keyboard_callback( const pcl::visualization::KeyboardEvent& event, void* )
{
                if ( event.keyDown () && event.getKeyCode () == 0x00000020) 
                  { 
                        pcl::io::savePLYFile("/home/long/digest.ply",*triangles_ptr);
                  } 
}

void PointCloudMapping::SetCurrentCameraPose(const cv::Mat &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}
