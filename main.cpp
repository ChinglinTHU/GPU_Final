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

void matches2points(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
        const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& pts_train,
        std::vector<Point2f>& pts_query);

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
	//	for (int i = 0; i < gray_frames.size(); i++)
		for (int i = 0; i < 20; i++)
		{
			printf("Computing %d frame feature\n", i);
			GpuMat cuda_frame;
			cuda_frame.upload(gray_frames[i]);
			GpuMat cur_points, cur_descriptors;
			surf(cuda_frame, GpuMat(), cur_points, cur_descriptors);
			keypointsGPU.push_back(cur_points);
			descriptorsGPU.push_back(cur_descriptors);
		}
		timer_count.Pause();
		printf_timer(timer_count);
		

		// model estimation
		int height = frames[0].rows;
		int width = frames[0].cols;
		int cut = 2*2*2;
		double weight = 1;
		
		vector<Mat> VecHomo;
		vector<Mat> VecImg;
    //	for (int i = 0; i < gray_frames.size()-1; i++)
		for (int i = 0; i < 11; i++)
		{
			printf("Computing %d and %d frame Homographies\n", i, i+1);
			vector<KeyPoint> keypoints1, keypoints2;
			surf.downloadKeypoints(keypointsGPU[i], keypoints1);
    		surf.downloadKeypoints(keypointsGPU[i+1], keypoints2);

			// match feature points
			vector<DMatch> matches, ransac_matches;
			vector<Point2f> prev_pts, now_pts;
			vector<unsigned char> match_mask;
   			matcher->match(descriptorsGPU[i], descriptorsGPU[i+1], matches);
   			matches2points(keypoints1, keypoints2, matches, prev_pts, now_pts);
   			for (int j = 0; j < prev_pts.size(); j++)
   			{
   				printf("(%.2f, %.2f) -> (%.2f, %.2f)\t\t", prev_pts[j].x, prev_pts[j].y)
   			}

   			// find global & pick out outliers
   			Mat globalHomo = findHomography(prev_pts, now_pts, RANSAC, 3, match_mask);
   			vector<Point2f> warp_pts;
   			perspectiveTransform(prev_pts, warp_pts, globalHomo);
   			printf("==========================\n");
   			printf("%.2f %.2f %.2f\n", globalHomo.at<float>(0, 0), globalHomo.at<float>(0, 1), globalHomo.at<float>(0, 2));
   			printf("%.2f %.2f %.2f\n", globalHomo.at<float>(1, 0), globalHomo.at<float>(1, 1), globalHomo.at<float>(1, 2));
   			printf("%.2f %.2f %.2f\n", globalHomo.at<float>(2, 0), globalHomo.at<float>(2, 1), globalHomo.at<float>(2, 2));
   			printf("==========================\n");
   			/*
   			for(int j = 0; j < match_mask.size(); j++)
   			{
   				printf("%.2f, ", norm(warp_pts[matches[j].trainIdx] - now_pts[matches[j].queryIdx]));
   			}
   			printf("\n");
   			*/

   			printf("sizeof match_mask = %d\n", match_mask.size());
   			for (int j = 0; j < match_mask.size(); j++)
   				if (match_mask[j] == 1)
   					ransac_matches.push_back(matches[j]);
   			printf("total ransac_matches = %d\n", ransac_matches.size());
   			allmatch.push_back(ransac_matches);

			asapWarp asap = asapWarp(height, width, cut+1, cut+1, 1); 
			asap.SetControlPts(keypoints1, keypoints2, allmatch[i]);
			asap.Solve();
			asap.PrintVertex();		
			
			// to get homographies for each cell of each frame
			Mat homo = Mat::zeros(cut, cut, CV_64FC(9));
			asap.CalcHomos(homo);
			VecHomo.push_back(homo);

			Mat img_matches;
    		drawMatches(Mat(frames[i]), keypoints1, Mat(frames[i+1]), keypoints2, allmatch[i], img_matches);
    		VecImg.push_back(img_matches);
		}

		imwrite("match_00.png", VecImg[0]);
		imwrite("match_01.png", VecImg[1]);
		imwrite("match_02.png", VecImg[2]);
		imwrite("match_03.png", VecImg[3]);
		imwrite("match_04.png", VecImg[4]);
		imwrite("match_05.png", VecImg[5]);
		imwrite("match_06.png", VecImg[6]);
		imwrite("match_07.png", VecImg[7]);
		imwrite("match_08.png", VecImg[8]);
		imwrite("match_09.png", VecImg[9]);
		imwrite("match_10.png", VecImg[10]);
		imwrite("frame_00.png", frames[0]);
		imwrite("frame_01.png", frames[1]);
		imwrite("frame_02.png", frames[2]);
		imwrite("frame_03.png", frames[3]);
		imwrite("frame_04.png", frames[4]);
		imwrite("frame_05.png", frames[5]);
		imwrite("frame_06.png", frames[6]);
		imwrite("frame_07.png", frames[7]);
		imwrite("frame_08.png", frames[8]);
		imwrite("frame_09.png", frames[9]);
		imwrite("frame_10.png", frames[10]);

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

void matches2points(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
        const vector<DMatch>& matches, vector<Point2f>& pts_train,
        vector<Point2f>& pts_query)
    {

        pts_train.clear();
        pts_query.clear();
        pts_train.reserve(matches.size());
        pts_query.reserve(matches.size());

        size_t i = 0;

        for (; i < matches.size(); i++)
        {

            const DMatch & dmatch = matches[i];

            pts_query.push_back(query[dmatch.queryIdx].pt);
            pts_train.push_back(train[dmatch.trainIdx].pt);

        }

    }
