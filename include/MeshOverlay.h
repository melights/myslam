#ifndef MESHOVERLAY_H
#define MESHOVERLAY_H


#include "System.h"
#include "Tracking.h"

#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkImageActor.h>
#include <vtkImageImport.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
using namespace ORB_SLAM2;

class MeshOverlay
{
public:

    std::mutex mMutex;
    vtkSmartPointer<vtkImageActor> imageActor;
    vtkSmartPointer<vtkRenderWindow> renderWindow;
    vtkSmartPointer<vtkRenderer> backgroundRenderer;
    vtkSmartPointer<vtkPolyData> poly;
    vtkSmartPointer<vtkPolyDataMapper> polymapper;
    vtkSmartPointer<vtkActor> polyactor;
    vtkSmartPointer<vtkRenderer> polyrenderer;
    vtkSmartPointer<vtkImageImport> importer;
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor;
    MeshOverlay(float fov,float Max_opa);
    int Init();
    void setImages(const cv::Mat &imLeft,const cv::Mat &inpose);
    void setMesh();
    void setCameraParameters(const Eigen::Matrix4f &extrinsics);
    void InitMat(cv::Mat &m, float *num);
    void grabImage();
    void setMesh(vtkSmartPointer< vtkPolyData > &poly_data);
private:
    shared_ptr<thread>  OverlayThread;   
    bool bindToImporter(cv::Mat &_src);
    float m_fov;
    float m_opa;    
    float m_Max_opa;
    float m_adder;
};

#endif // POINTCLOUDMAPPING_H
