#include "./../../local/include/opencv2/core.hpp"
#include "./../../local/include/opencv2/videostab.hpp"
#include "./../../local/include/opencv2/core/cuda.hpp"
#include "./../../local/include/opencv2/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::videostab;

int main()
{
    std::string videoFile = "corridor.MOV";
    // int r = cv::cuda::getCudaEnabledDeviceCount();
    // cout << r << endl;
    MotionModel model = cv::videostab::MM_TRANSLATION; //Type of motion to compensate
    bool use_gpu = true; //Select CUDA version or "regular" version

    cv::Ptr<VideoFileSource> video = cv::makePtr<VideoFileSource>(videoFile,true);
    cv::Ptr<OnePassStabilizer> stabilizer = cv::makePtr<OnePassStabilizer>();

    cv::Ptr<MotionEstimatorBase> MotionEstimator = cv::makePtr<MotionEstimatorRansacL2>(model);

    cv::Ptr<ImageMotionEstimatorBase> ImageMotionEstimator;

    if (use_gpu)
    {
    	cout << "Im using GPU!!!" << endl;
        ImageMotionEstimator = cv::makePtr<KeypointBasedMotionEstimatorGpu>(MotionEstimator);
    }
    else
    {
    	cout << "fuck off." << endl;
        ImageMotionEstimator = cv::makePtr<KeypointBasedMotionEstimator>(MotionEstimator);
    }

    stabilizer->setFrameSource(video);
    stabilizer->setMotionEstimator(ImageMotionEstimator);
    stabilizer->setLog(cv::makePtr<cv::videostab::NullLog>()); //Disable internal prints

    string windowtTitle = "Stabilized Video";
    cv::namedWindow(windowtTitle, cv::WINDOW_AUTOSIZE);

    int cur_frame = 1;
    while(true)
    {
    	cur_frame++;
        cv::Mat frame = stabilizer->nextFrame();
        if(frame.empty())   break;
        cv::imshow(windowtTitle, frame);
        cv::waitKey(10);
    }
    cout << "total frame: " << cur_frame << endl;

    return 0;
}
