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

#include "./mesh/asapWarp.h"
#include "./mesh/mesh.h"
#include "./mesh/quad.h"
#include "./utils/Timer.h"

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
 	   		cuda::printShortCudaDeviceInfo(cuda::getDevice());
	
		string inputPath = arg(0);
		if (inputPath.empty())
			throw runtime_error("pls specify video file path");
		
		// get original video and its frames
		Ptr<VideoFileSource> source = makePtr<VideoFileSource>(inputPath);
		cout << "total frame: " << source->count() << endl;
		if(source->count() == 0)
			return 0;
		outputFps = source->fps();
		vector<Mat> frames, gray_frames;
		while(true)
		{
			Mat cur_frame, gray_frame;
			cur_frame = source->nextFrame();
			if(cur_frame.empty()) break;
			cv::cvtColor(cur_frame, gray_frame, CV_BGR2GRAY);
			frames.push_back(cur_frame); // TODO: do i need to change its format?
			gray_frames.push_back(gray_frame);
		}

		// feature detect on GPU
		vector<GpuMat> keypointsGPU, descriptorsGPU;
		vector<vector<DMatch> > allmatch;
		SURF_CUDA surf; // TODO: using SURF is a little slow, should change its params or change a way (Orb, FAST, BRIEF)
		Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
		surf.hessianThreshold = 5000;
		Timer timer_count;
		timer_count.Start();
		//for (int i = 0; i < gray_frames.size(); i++)
		for (int i = 0; i < 2; i++)
		{
			GpuMat cuda_frame;
			cuda_frame.upload(gray_frames[i]);
			GpuMat cur_points, cur_descriptors;
			surf(cuda_frame, GpuMat(), cur_points, cur_descriptors);
			keypointsGPU.push_back(cur_points);
			descriptorsGPU.push_back(cur_descriptors);
			
			if (i > 0)
			{
				vector<DMatch> matches;
    			matcher->match(descriptorsGPU[i-1], descriptorsGPU[i], matches);
    			allmatch.push_back(matches);
			}

			printf("points: rows: %d, cols: %d\n", cur_points.rows, cur_points.cols);
			printf("descrip: rows: %d, cols: %d\n", cur_descriptors.rows, cur_descriptors.cols);
		}
		timer_count.Pause();
		printf_timer(timer_count);
		

		// model estimation
		int height = frames[0].rows;
		int width = frames[0].cols;
		int cut = 2*2*2;
		double weight = 1;
		asapWarp asap = asapWarp(height, width, cut+1, cut+1, 1); 
		//asap.PrintConstraints();

		vector<KeyPoint> keypoints1, keypoints2;
		surf.downloadKeypoints(keypointsGPU[0], keypoints1);
    	surf.downloadKeypoints(keypointsGPU[1], keypoints2);

    	/*
		Mat img_matches;
	    drawMatches(frames[0], keypoints1, frames[1], keypoints2, allmatch[0], img_matches);

	    namedWindow("matches", 0);
	    imshow("matches", img_matches);
	    waitKey(0);
	    */

		asap.SetControlPts(keypoints1, keypoints2, allmatch[0]);
		// asap.PrintConstraints(true);
		asap.Solve();
		asap.PrintVertex();
		// to get homographies for each cell of each frame

		// bundled camera path

		// path optimization
	}
	catch (const exception &e)
	{
		cout << "error: " << e.what() << endl;
		return -1;
	}

    return 0;
}
