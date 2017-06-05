#include <string>
#include <iostream>
#include <stdexcept>
#include <vector>

// cv libraries
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videostab.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/opencv_modules.hpp"

#define arg(name) cmd.get<string>(name)
#define argb(name) cmd.get<bool>(name)
#define argi(name) cmd.get<int>(name)
#define argf(name) cmd.get<float>(name)
#define argd(name) cmd.get<double>(name)

using namespace std;
using namespace cv;
using namespace cv::videostab;
using namespace cv::cuda;

Ptr<IFrameSource> stabilizedFrames;
string saveMotionsPath;
double outputFps;
string outputPath;
bool quietMode;

int main(int argc, const char **argv)
{
	try
	{
		const char *keys = 
			"{@1            |        | }" // fuck you
			"{model         | affine | }"
			"{gpu           | yes    | }";
		CommandLineParser cmd(argc, argv, keys);

    	if (arg("gpu") == "yes")
 	    {
 	   		cuda::printShortCudaDeviceInfo(cuda::getDevice());
    	}
	
		string inputPath = arg(0);
		if (inputPath.empty())
			throw runtime_error("specify video file path");
		
		// get original video
		Ptr<VideoFileSource> source = makePtr<VideoFileSource>(inputPath);
		cout << "total frame: " << source->count() << endl;
		outputFps = source->fps();

		// feature detect
		vector<GpuMat> keypointsGPU, descriptorsGPU;
		vector<Mat> frames;
		SURF_CUDA surf;
		surf.hessianThreshold = 10000;
		Mat tmp_img;
		while(true)
		{
			Mat cur_frame;
			GpuMat cuda_frame;
			cur_frame = source->nextFrame();
			if (cur_frame.empty()) break;
			cv::cvtColor(cur_frame, tmp_img, CV_BGR2GRAY);
			cuda_frame.upload(cur_frame);
			frames.push_back(tmp_img);
			GpuMat cur_points, cur_descriptors;
			surf(cuda_frame, GpuMat(), cur_points, cur_descriptors);
			keypointsGPU.push_back(cur_points);
			descriptorsGPU.push_back(cur_descriptors);
		}

		// test
		//matching descriptors
    	Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
    	vector<DMatch> matches;
    	matcher->match(descriptorsGPU[0], descriptorsGPU[1], matches);

    	// downloading results
    	vector<KeyPoint> keypoints1, keypoints2;
    	vector<float> descriptors1, descriptors2;
    	surf.downloadKeypoints(keypointsGPU[0], keypoints1);
    	surf.downloadKeypoints(keypointsGPU[1], keypoints2);
    	surf.downloadDescriptors(descriptorsGPU[0], descriptors1);
    	surf.downloadDescriptors(descriptorsGPU[1], descriptors2);

    	// find Homography
    	vector<Point2f> src, dst;
    	for (int i = 0; i < matches.size(); i++)
    	{
    		src.push_back(keypoints1[matches[i].queryIdx].pt);
    		dst.push_back(keypoints2[matches[i].trainIdx].pt);
    	}
    	Mat H = findHomography(src, dst, CV_RANSAC);
    	cout << "M = " << endl << " " << H << endl << endl;
    	// Mat imgout;
    	// warpPerspective(frames[0], imgout, H, frames[0].size());
    	// imshow("src", frames[0]);
    	// imshow("dst", frames[1]);
    	// imshow("warp", imgout);


    	// draw the matched img
    	Mat img_matches;
    	drawMatches(Mat(frames[0]), keypoints1, Mat(frames[1]), keypoints2, matches, img_matches);

    	// namedWindow("matches", 0);
    	// imshow("matches", img_matches);
    	waitKey(0);
	}
	catch (const exception &e)
	{
		cout << "error: " << e.what() << endl;
		return -1;
	}

    return 0;
}
