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

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include "System.h"
#include "Tracking.h"

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <condition_variable>
#include <pcl/visualization/pcl_visualizer.h>
using namespace ORB_SLAM2;

class PointCloudMapping
{
public:
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    typedef pcl::PointXYZ PointTmono;
    typedef pcl::PointCloud<PointTmono> PointCloudmono;
    
    PointCloudMapping( double resolution_ );
    
    // 插入一个keyframe，会更新一次地图
    void insertKeyFrame(KeyFrame* kf, cv::Mat& left_img, cv::Mat& right_img);
    void shutdown();
    void Cloud_Viewer();
    void Surface_Viewer();
    void update();
void keyboard_callback( const pcl::visualization::KeyboardEvent& event, void* );
void SetCurrentCameraPose(const cv::Mat &Tcw);

protected:
    pcl::PointCloud< PointCloudMapping::PointT >::Ptr generatePointCloud();
    pcl::PointCloud< PointCloudMapping::PointTmono >::Ptr generatePointCloudmono();
    
    PointCloudmono::Ptr globalMap;
    shared_ptr<thread>  cloudViewerThread;   
    shared_ptr<thread>  surfaceViewerThread;   
    shared_ptr<thread>  updateThread;
    Map* mpMap;
    bool    shutDownFlag    =false;
    mutex   shutDownMutex;  
    
    condition_variable  keyFrameUpdated;
    mutex               keyFrameUpdateMutex;
    
    // data to generate point clouds
    vector<KeyFrame*>       keyframes;
    KeyFrame*       keyframe;
    cv::Mat         left;
    cv::Mat         right;
    
    cv::Mat         depthImgs;
    mutex                   keyframeMutex;
    mutex                   reconstructionMutex;
    uint16_t                lastKeyframeSize =0;
    
     double resolution = 0.04;
    pcl::VoxelGrid<PointTmono>  voxel;
    pcl::PolygonMesh::Ptr triangles_ptr;
    cv::Mat mCameraPose;

    std::mutex mMutexCamera;

};

#endif // POINTCLOUDMAPPING_H
