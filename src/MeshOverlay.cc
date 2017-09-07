// Include standard headers
#include <stdio.h>
#include <vector>

// Include OpenCV
#include <opencv2/opencv.hpp>
#include <MeshOverlay.h>
//#include <vtkVersion.h>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkImageActor.h>
#include <vtkImageData.h>
#include <vtkImageImport.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>
#include "vtkOBJReader.h"

#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>

#include <vtkPLYReader.h>
#include <vtkSphereSource.h>
#include <vtkProperty.h>
#include <vtkPropPicker.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>

#include <vtkSmartPointer.h>
#include <vtkSuperquadricSource.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>
#include <vtkMath.h>
#include <time.h>
#include <vtkActorCollection.h>
#include <vtkRendererCollection.h>
#include <vtkPolygonalSurfacePointPlacer.h>
#include <vtkOrientedGlyphContourRepresentation.h>
#include <vtkContourWidget.h>
#include <vtkPolyDataCollection.h>

using namespace cv;




class vtkTimerCallback2 : public vtkCommand
{
  public:
    static vtkTimerCallback2 *New()
    {
        vtkTimerCallback2 *cb = new vtkTimerCallback2;
        cb->TimerCount = 0;
        return cb;
    }

    virtual void Execute(vtkObject *caller, unsigned long eventId,
                         void *vtkNotUsed(callData))
    {
        vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::SafeDownCast(caller);
        if (vtkCommand::TimerEvent == eventId)
        {
            ++this->TimerCount;
        }
        iren->GetRenderWindow()->Render();
    }

  private:
    int TimerCount;
    
};

class MouseInteractorStyle2 : public vtkInteractorStyleTrackballCamera
{
  public:
    static MouseInteractorStyle2* New();
    vtkTypeMacro(MouseInteractorStyle2, vtkInteractorStyleTrackballCamera);

    virtual void OnRightButtonDown()
    {
      int* clickPos = this->GetInteractor()->GetEventPosition();
	
      // Pick from this location.
      vtkSmartPointer<vtkPropPicker>  picker =
        vtkSmartPointer<vtkPropPicker>::New();
      picker->Pick(clickPos[0], clickPos[1], 0, this->GetDefaultRenderer());
      double* pos = picker->GetPickPosition();
      std::cout << "Pick position (world coordinates) is: "
                << pos[0] << " " << pos[1]
                << " " << pos[2] << std::endl;

      std::cout << "Picked actor: " << picker->GetActor() << std::endl;
      //Create a sphere
    //   vtkSmartPointer<vtkSphereSource> sphereSource =
    //     vtkSmartPointer<vtkSphereSource>::New();
    //   sphereSource->SetCenter(pos[0], pos[1], pos[2]);
    //   sphereSource->SetRadius(0.01);
if(pos[0]==0||pos[1]==0||pos[2]==0){
    vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
    return;
}
  vtkSmartPointer<vtkOBJReader> reader =
    vtkSmartPointer<vtkOBJReader>::New();
  reader->SetFileName("/home/long/Meshoverlay/arrow.obj");
  reader->Update();
vtkSmartPointer<vtkTransform> transform =
    vtkSmartPointer<vtkTransform>::New();
  transform->Scale(3,3,3);
 
  vtkSmartPointer<vtkTransformFilter> transformFilter =
    vtkSmartPointer<vtkTransformFilter>::New();
  transformFilter->SetInputConnection(reader->GetOutputPort());
  transformFilter->SetTransform(transform);
      //Create a mapper and actor
      vtkSmartPointer<vtkPolyDataMapper> mapper =
        vtkSmartPointer<vtkPolyDataMapper>::New();
      mapper->SetInputConnection(transformFilter->GetOutputPort());

      vtkSmartPointer<vtkActor> actor =
        vtkSmartPointer<vtkActor>::New();
      actor->SetMapper(mapper);
      actor->SetPosition(pos[0], pos[1], pos[2]);
      vtkMath::RandomSeed(time(NULL));
      double R = vtkMath::Random(0.0,1.0);
      double G = vtkMath::Random(0.0,2.0);
      double B = vtkMath::Random(0.0,2.0);
      actor->GetProperty()->SetColor(R, G, B); //(R,G,B)

      //this->GetInteractor()->GetRenderWindow()->GetRenderers()->GetDefaultRenderer()->AddActor(actor);
      this->GetDefaultRenderer()->AddActor(actor);
      
      // Forward events
      vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
    }

  private:

};
vtkStandardNewMacro(MouseInteractorStyle2);
MeshOverlay::MeshOverlay(float fov, float Max_opa):m_fov(fov),m_opa(Max_opa),m_Max_opa(Max_opa),m_adder(Max_opa/25),init(false)
{
    //frameL=image.copy();
    OverlayThread = make_shared<thread>(bind(&MeshOverlay::Init, this));
    
}
void MeshOverlay::InitMat(Mat &m, float *num)
{
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            m.at<float>(j, i) = *(num + i * m.rows + j);
}

void MeshOverlay::setCameraParameters(const Eigen::Matrix4f &extrinsics)
{
    // Position = extrinsic translation
    Eigen::Vector3f pos_vec = extrinsics.block<3, 1>(0, 3);

    // Rotate the view vector
    Eigen::Matrix3f rotation = extrinsics.block<3, 3>(0, 0);
    Eigen::Vector3f y_axis(0.f, 1.f, 0.f);
    Eigen::Vector3f up_vec(rotation * y_axis);

    // Compute the new focal point
    Eigen::Vector3f z_axis(0.f, 0.f, 1.f);
    Eigen::Vector3f focal_vec = pos_vec + rotation * z_axis;

    vtkSmartPointer<vtkCamera> cam = polyrenderer->GetActiveCamera();
    cam->SetViewAngle( m_fov );
    cam->SetWindowCenter(0, 0);
    cam->SetPosition(pos_vec[0], pos_vec[1], pos_vec[2]);
    cam->SetFocalPoint(focal_vec[0], focal_vec[1], focal_vec[2]);
    cam->SetViewUp(up_vec[0], up_vec[1], up_vec[2]);

}

bool MeshOverlay::bindToImporter(cv::Mat &_src)
{

    importer->SetDataSpacing(1, 1, 1);
    importer->SetDataOrigin(0, 0, 0);
    importer->SetWholeExtent(0, _src.size().width - 1, 0, _src.size().height - 1, 0, 0);
    importer->SetDataExtentToWholeExtent();
    importer->SetDataScalarTypeToUnsignedChar();
    importer->SetNumberOfScalarComponents(_src.channels());
    importer->SetImportVoidPointer(_src.data);
    importer->Update();
    return true;
}

void MeshOverlay::setImages(const cv::Mat &imLeft,const cv::Mat &inpose)
{
    cv::Mat pose=inpose.clone();
    unique_lock<mutex> lock(mMutex);
    Mat tmp1;
    cv::cvtColor(imLeft, tmp1, CV_BGR2RGB);
    cv::flip(tmp1, frameL, 0);
    //grabImage();
    importer->Modified();
    importer->Update(); 
    if (pose.rows == 0){
        return;
    }
    Mat R, tvec;
    pose.rowRange(0, 3).colRange(0, 3).copyTo(R);
    pose.rowRange(0, 3).col(3).copyTo(tvec);
    float qx, qy, qz, qw;
    float px, py, pz, pw;

    qw = sqrt(1.0 + R.at<float>(0, 0) + R.at<float>(1, 1) + R.at<float>(2, 2)) / 2.0;
    qx = (R.at<float>(2, 1) - R.at<float>(1, 2)) / (4 * qw);
    qy = -(R.at<float>(0, 2) - R.at<float>(2, 0)) / (4 * qw);
    qz = -(R.at<float>(1, 0) - R.at<float>(0, 1)) / (4 * qw);

    pw = -qz;
    px = qy;
    py = qx;
    pz = qw;

    qw = pw;
    qx = px;
    qy = py;
    qz = pz;

    float m0[] = {1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy + 2 * qz * qw, 2 * qx * qz - 2 * qy * qw, 0, 2 * qx * qy - 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz + 2 * qx * qw, 0, 2 * qx * qz + 2 * qy * qw, 2 * qy * qz - 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy, 0, -tvec.at<float>(0, 0), -tvec.at<float>(1, 0), -tvec.at<float>(2, 0), 1};
    MeshOverlay::InitMat(pose, m0);

    Eigen::Matrix4f extrinsics;

    extrinsics << pose.at<float>(0, 0), pose.at<float>(0, 1), pose.at<float>(0, 2), pose.at<float>(0, 3),
        pose.at<float>(1, 0), pose.at<float>(1, 1), pose.at<float>(1, 2), pose.at<float>(1, 3),
        pose.at<float>(2, 0), pose.at<float>(2, 1), pose.at<float>(2, 2), pose.at<float>(2, 3),
        pose.at<float>(3, 0), pose.at<float>(3, 1), pose.at<float>(3, 2), pose.at<float>(3, 3);
    //std::cout << extrinsics << std::endl;
    setCameraParameters(extrinsics);
    //renderWindow->Render();
    //std::cout<<"rendered!!"<<std::endl;
    if(m_opa>m_Max_opa||m_opa<=0.01)
        m_adder=-m_adder;
    m_opa+=m_adder;
    polyactor->GetProperty()->SetOpacity(m_opa);
}

void MeshOverlay::setMesh(vtkSmartPointer< vtkPolyData > &poly_data)
{
    std::cout<<"received polydata"<<std::endl;
    poly->DeepCopy(poly_data);
    if(!init){
        pilotrenderer->ResetCamera();
        init=true;
    }

}
int MeshOverlay::Init()
{
    //Set playback
    //grabImage();
    //frameL=image.copy();
    importer = vtkSmartPointer<vtkImageImport>::New();
    frameL = Mat::zeros( 480, 640, CV_8UC3 );
    bindToImporter(frameL);

    imageActor = vtkSmartPointer<vtkImageActor>::New();
    imageActor->SetInput(importer->GetOutput());

    backgroundRenderer = vtkSmartPointer<vtkRenderer>::New();
    backgroundRenderer->SetLayer(0);
    backgroundRenderer->InteractiveOff();
    backgroundRenderer->SetViewport(0.0, 0.0, 0.5, 1.0);


    /////////////////// Polydata /////////////////////
    polyrenderer = vtkSmartPointer<vtkRenderer>::New();
    polyrenderer->SetViewport(0.0, 0.0, 0.5, 1.0);
    //vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
    ////reader->SetFileName("/home/long/digest.ply");
    //reader->SetFileName("../digest.ply");
    
    poly = vtkSmartPointer<vtkPolyData>::New();
    // poly->DeepCopy(reader->GetOutput());
    
    polymapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    polymapper->SetInput(poly);
    //mapper->SetScalarVisibility(0);

    
    polyactor = vtkSmartPointer<vtkActor>::New();
    polyactor->SetMapper(polymapper);
    polyactor->GetProperty()->SetRepresentationToWireframe();
    polyactor->GetProperty()->SetOpacity(0.4);
    // polyactor->GetProperty()->SetRenderLinesAsTubes(1);
    // polyactor->GetProperty()->SetRenderPointsAsSpheres(1);
    // polyactor->GetProperty()->SetVertexVisibility(1);
    // polyactor->GetProperty()->SetVertexColor(0.5,1.0,0.8);
    polyrenderer->SetLayer(1);
    //polyrenderer->SetUseShadows(0);
    //polyrenderer->SetOcclusionRatio(1);
    /////////////////// PilotView /////////////////////

    pilotrenderer = vtkSmartPointer<vtkRenderer>::New();
    pilotrenderer->SetViewport(0.5, 0.0, 1.0, 1.0);
    //vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
    ////reader->SetFileName("/home/long/digest.ply");
    //reader->SetFileName("../digest.ply");
    
    vtkSmartPointer<vtkActor> pilotactor = vtkSmartPointer<vtkActor>::New();
    pilotactor->SetMapper(polymapper);
    pilotactor->GetProperty()->SetRepresentationToWireframe();

    // polyactor->GetProperty()->SetRenderLinesAsTubes(1);
    // polyactor->GetProperty()->SetRenderPointsAsSpheres(1);
    // polyactor->GetProperty()->SetVertexVisibility(1);
    // polyactor->GetProperty()->SetVertexColor(0.5,1.0,0.8);
    //pilotrenderer->SetLayer(0);


/////////////////// Render Window /////////////////////


    renderWindow =
        vtkSmartPointer<vtkRenderWindow>::New();


    renderWindow->SetSize(1280,480);
    renderWindow->SetNumberOfLayers(2);
    renderWindow->AddRenderer(backgroundRenderer);
    renderWindow->AddRenderer(polyrenderer);
    renderWindow->AddRenderer(pilotrenderer);
    //renderWindow->AddRenderer(sphererenderer);

    renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Add actors to the renderers
    polyrenderer->AddActor(polyactor);
    pilotrenderer->AddActor(pilotactor);
    backgroundRenderer->AddActor(imageActor);
    // sphererenderer->AddActor(sphereactor);
    // Render once to figure out where the background camera will be
    renderWindow->Render();
///////////////////////////////////////////
  vtkSmartPointer<vtkContourWidget> contourWidget = 
    vtkSmartPointer<vtkContourWidget>::New();
  contourWidget->SetInteractor(renderWindowInteractor);
    
  vtkOrientedGlyphContourRepresentation* rep = 
    vtkOrientedGlyphContourRepresentation::SafeDownCast(
    contourWidget->GetRepresentation());

  vtkSmartPointer<vtkPolygonalSurfacePointPlacer> pointPlacer =
    vtkSmartPointer<vtkPolygonalSurfacePointPlacer>::New();
  pointPlacer->AddProp(polyactor);
  pointPlacer->AddProp(pilotactor);
  pointPlacer->GetPolys()->AddItem(polymapper->GetInput());

  rep->GetLinesProperty()->SetColor(1, 0, 0);
  rep->GetLinesProperty()->SetLineWidth(5.0);
  rep->SetPointPlacer(pointPlacer);
  
  contourWidget->EnabledOn();

/////////////////// Background Camera /////////////////////

    double origin[3];
    double spacing[3];
    int extent[6];
    importer->GetOutput()->GetOrigin(origin);
    importer->GetOutput()->GetSpacing(spacing);
    importer->GetOutput()->GetExtent(extent);

    vtkCamera *camera = backgroundRenderer->GetActiveCamera();
    camera->ParallelProjectionOn();

    double xc = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0];
    double yc = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1];
    double yd = (extent[3] - extent[2] + 1) * spacing[1];
    double d = camera->GetDistance();
    camera->SetParallelScale(0.5 * yd);
    camera->SetFocalPoint(xc, yc, 0.0);
    camera->SetPosition(xc, yc, d);

    renderWindow->Render();

/////////////////// Interaction /////////////////////
      vtkSmartPointer<MouseInteractorStyle2> style =
    vtkSmartPointer<MouseInteractorStyle2>::New();
  style->SetDefaultRenderer(polyrenderer);

  renderWindowInteractor->SetInteractorStyle( style );

    renderWindowInteractor->Initialize();
/////////////////// Timer /////////////////////

    vtkSmartPointer<vtkTimerCallback2> cb = vtkSmartPointer<vtkTimerCallback2>::New();
    renderWindowInteractor->AddObserver(vtkCommand::TimerEvent, cb);

    int timerId = renderWindowInteractor->CreateRepeatingTimer(30);
    std::cout << "timerId: " << timerId << std::endl;
    // Start the interaction and timer
    renderWindowInteractor->Start();

    return 0;
}
