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

#include "opencv2/stitching/detail/blenders.hpp"

#include "./mesh/warp.h"
#include "./mesh/asapWarp.h"
#include "./path/allPath.h"
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
using namespace cv::detail;

typedef Vec<float, 9> Vec9f;
typedef Vec<double, 9> Vec9d;

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
		int s = 0;
		while(true)
		{
			Mat cur_frame, gray_frame;
			cur_frame = source->nextFrame();
			if(cur_frame.empty()) break;
			cv::cvtColor(cur_frame, gray_frame, CV_BGR2GRAY);
			// if (s == 8 || s == 50 || s == 100 || s == 150 || s == 200 || s == 246)
			if (true)
			{
				frames.push_back(cur_frame); // TODO: do i need to change its format?
				gray_frames.push_back(gray_frame);
			}
			s++;
		}

		// feature detect on GPU
		vector<GpuMat> keypointsGPU, descriptorsGPU;
		vector<vector<Point2f>> vec_now_pts, vec_next_pts;
		vector<Mat> vec_global_homo;
		SURF_CUDA surf; // TODO: using SURF is a little slow, should change its params or change a way (Orb, FAST, BRIEF)
		Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
		surf.hessianThreshold = 5000;
		Timer timer_count;
		timer_count.Start();
		
		for (int i = 0; i < gray_frames.size(); i++)
	//	for (int i = 0; i < 100; i++)
		{
			printf("Detecting feature: %d \n", i);
			GpuMat cuda_frame;
			cuda_frame.upload(gray_frames[i]);
			GpuMat cur_points, cur_descriptors;
			surf(cuda_frame, GpuMat(), cur_points, cur_descriptors);
			keypointsGPU.push_back(cur_points);
			descriptorsGPU.push_back(cur_descriptors);

			if (i > 0)
			{
				// match feature points
				vector<KeyPoint> keypoints1, keypoints2;
				surf.downloadKeypoints(keypointsGPU[i-1], keypoints1);
	    		surf.downloadKeypoints(keypointsGPU[i], keypoints2);

	    		vector<DMatch> matches, ransac_matches;
				vector<Point2f> now_pts, next_pts;
				vector<unsigned char> match_mask;
	   			matcher->match(descriptorsGPU[i-1], descriptorsGPU[i], matches);
	   			matches2points(keypoints1, keypoints2, matches, now_pts, next_pts);

	   			// find global & pick out outliers
	   			// findHomography return mat with double type
	   			Mat globalHomo = findHomography(now_pts, next_pts, RANSAC, 4, match_mask);
	   			for (int j = match_mask.size() - 1; j >= 0; j--)
	   				if (match_mask[j] == 0)
	   				{
	   					now_pts.erase(now_pts.begin()+j);
	   					next_pts.erase(next_pts.begin()+j);
	   				}
	   			if(now_pts.size() != next_pts.size())
	   			   	throw runtime_error("matching points have different size\n");
	   			printf("\t(ransac_num = %d)\n", now_pts.size());
	   			vec_now_pts.push_back(now_pts);
	   			vec_next_pts.push_back(next_pts);
	   			vec_global_homo.push_back(globalHomo);
   			}
		}
		timer_count.Pause();
		printf_timer(timer_count);
		
		// model estimation
		int height = frames[0].rows;
		int width = frames[0].cols;
		int cut = 2*2*2;
		double weight = 1;
		
		vector<Mat> VecHomo;
		
		for (int i = 0; i < vec_now_pts.size(); i++)
		{
			asapWarp asap = asapWarp(height, width, cut+1, cut+1, 1); 
			printf("Computing frame Homographies (%d, %d) \n", i, i+1);	

			asap.SetControlPts(vec_now_pts[i], vec_next_pts[i], vec_global_homo[i]);
			//cerr << "Solve()" << endl;
			asap.Solve();
			// asap.PrintVertex();		

			//cerr << "CalcHomos()" << endl;
			// to get homographies for each cell of each frame
			Mat homo = Mat::zeros(cut, cut, CV_32FC(9));
			asap.CalcHomos(homo);
			//cerr << "end" << endl;
			VecHomo.push_back(homo);
			
		}

		// Compute bundled camera path
		vector<Mat> Vec;
		
		allPath allpath = allPath(cut, cut, VecHomo.size()+1);
		Mat homo = Mat::eye(3, 3, CV_32FC1);
		for (int t = 0; t < VecHomo.size(); t++)
		{
			printf("Compute bundled camera path %d \n", t);	
			for (int i = 0; i < cut; i++)
				for (int j = 0; j < cut; j++)
				{
					Vec9f tmp = VecHomo[t].at<Vec9f>(i, j);
					allpath.setHomo(i, j, t, tmp);
				}

		}
		
		allpath.computePath();

		allpath.optimizePath(20);
		// allpath.jacobiSolver();

		allpath.computeWarp();

		vector<Mat> path = allpath.getPath(0, 0);
		vector<Mat> optpath = allpath.getOptimizedPath(0, 0);
		Mat picture(1000, 1000, CV_8UC3, Scalar(255,255,255));  
		vector<Point2f> center(1);
		vector<Point2f> move(1);
		vector<Point2f> stable(1);
		vector<Point2f> tmp(1);
		
		float scale = 1.f;
		Point2f offset(500.f, 500.f);
		for (int i = 0; i < path.size(); i++)
		{
			if (i == 0)
			{
				center[0] = Point2f(10.f, 10.f);
				move[0]   = scale*center[0] + offset;
				stable[0] = scale*center[0] + offset;
			}
			else
			{
				tmp[0] = move[0];
				perspectiveTransform(center, move, path[i]);
				move[0] = move[0]*scale + offset;
				arrowedLine(picture, tmp[0], move[0], Scalar(255,0,0));  // blue 
				tmp[0] = stable[0];
				perspectiveTransform(center, stable, optpath[i]);
				stable[0] = stable[0]*scale + offset;
				arrowedLine(picture, tmp[0], stable[0], Scalar(0,0,255));  // red
			}
		}
		imwrite("optimize_path.png", picture);
		
		//namedWindow("Display window", WINDOW_AUTOSIZE);
		//imshow("Display window", picture );
		//waitKey(0);

		

		// Warp image
		vector<Mat> warp_frames;
		Mat globalH = Mat::eye(3, 3, CV_64FC1);
		
		asapWarp asap = asapWarp(height, width, cut+1, cut+1, 1); 
		warp W(asap);
		for (int t = 0; t < min(1000, allpath.time); t++)
		{
			//if (t != 90)
			//	continue;
			
			printf("Image Synthesis %d \n", t);
			///* my new warpimg method
			Mat warp_frame;
			W.warpImageMesh(frames[t], warp_frame, allpath.getPath(t), allpath.getOptimizedPath(t));
			// warp_frames.push_back(warp_frame);
			//cerr << "imagesyn: 1" << endl;

			//cerr << warp_frame.size() << " warp_frame.size() " << endl;
			//cerr << frames[t].size() << " frames[t].size() " << endl;

			printf("Write images %d \n", t);
			/* write images */
			char str[20];
			Mat frame_warp = Mat::zeros(frames[t].rows + warp_frame.rows, frames[t].cols, CV_8UC3);
			frames[t].copyTo(frame_warp(Rect(0, 0, frames[t].cols, frames[t].rows)));
			warp_frame.copyTo(frame_warp(Rect(0, frames[t].rows, warp_frame.cols, warp_frame.rows)));

			sprintf(str, "result/frame_warp_%03d.png", t);
			imwrite(str, frame_warp);

			//*/

			/* original method
			Ptr<Blender> blender;
			blender = Blender::createDefault(Blender::FEATHER, true);
			FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
			fb->prepare(Rect(0, 0, width, height));
			
			if (t > 0)
				globalH = vec_global_homo[t-1] * globalH;

			cout << "global at time " << t << endl;
			cout << globalH << endl;
			
			Mat frame = frames[t];
			frame.convertTo(frame, CV_16SC3);
			for (int i = 0; i < cut; i++)
				for (int j = 0; j < cut; j++)
				{
					Mat warp_frame, mask, warp_mask, h;
					h = allpath.getWarpHomo(i, j, t); // PC^(-1)
					// h = allpath.getPath(i, j, t).inv(); // C^(-1)
					// h = globalH.inv(); // G^(-1)


					mask = Mat::zeros(frame.size(), CV_8U);
					mask(Rect(asap.compute_pos(i, j), asap.compute_pos(i+1, j+1))).setTo(Scalar::all(255));

					warpPerspective(frame, warp_frame, h, frame.size());
					warpPerspective(mask, warp_mask, h, mask.size());

					fb->feed(warp_frame, warp_mask, Point(0, 0));
				}

			Mat warp_frame;
			Mat mask = Mat::zeros(frame.size(), CV_8U);
			mask.setTo(Scalar::all(255));
			fb->blend(warp_frame, mask);
			warp_frame.convertTo(warp_frame, CV_8UC3);

			warp_frames.push_back(warp_frame);
			*/
		}

		/*
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
		*/

	}
	catch (const exception &e)
	{
		cout << "error: " << e.what() << endl;
		return -1;
	}

    return 0;
}

void matches2points(const vector<KeyPoint>& query, const vector<KeyPoint>& train, 
        const vector<DMatch>& matches, vector<Point2f>& pts_query, vector<Point2f>& pts_train)
    {

        pts_train.clear();
        pts_query.clear();
        pts_train.reserve(matches.size());
        pts_query.reserve(matches.size());

        size_t i = 0;

        for (; i < matches.size(); i++)
        {
            const DMatch & dmatch = matches[i];
            Point2f q = query[dmatch.queryIdx].pt;
            Point2f t = train[dmatch.trainIdx].pt;

            pts_query.push_back(q);
            pts_train.push_back(t);

        }

    }


