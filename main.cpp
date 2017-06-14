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
			// if (s < 50)
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
		int cuth = cut;
		int cutw = cut;
		float quadWidth = width/cut;
		float quadHeight = height/cut;
		double weight = 1;
		
		vector<BundleHomo> VecHomo;
		vector<vector<vector<Point2f>>> allCellPoints; // t,i,j
		vector<vector<Point2f>> preCellPoints(cut+1, vector<Point2f> (cut+1));
		vector<vector<Point2f>> curCellPoints(cut+1, vector<Point2f> (cut+1));
		for(int i = 0; i<cut+1;i++)
			for(int j = 0;j<cut+1;j++)
			{
				Point2f p = Point2f(float(i*quadWidth), float(j*quadHeight));
				preCellPoints[i][j] = p;
			}
		allCellPoints.push_back(preCellPoints);
		
		timer_count.Reset();
		timer_count.Start();		
		for (int i = 0; i < vec_now_pts.size(); i++)
		{
			asapWarp asap = asapWarp(height, width, cuth+1, cutw+1, 1); 
			printf("Computing frame Homographies (%d, %d) \n", i, i+1);	

			//asap.SetControlPts(vec_now_pts[i], vec_next_pts[i], vec_global_homo[i]);
			asap.SetControlPts(vec_next_pts[i], vec_now_pts[i], vec_global_homo[i].inv());
			//cerr << "Solve()" << endl;
			asap.Solve();
			// asap.PrintVertex();		
			asap.SolvePoints(preCellPoints, curCellPoints);
			allCellPoints.push_back(curCellPoints);
			preCellPoints = curCellPoints;

			//cerr << "CalcHomos()" << endl;
			// to get homographies for each cell of each frame
			// BundleHomo homo; // = Mat::zeros(cutw, cuth, CV_32FC(9));
			BundleHomo homo_t1_t0(width-1, vector<Mat> (height-1));
			asap.CalcHomos(homo_t1_t0);
			//cerr << "end" << endl;
			VecHomo.push_back(homo_t1_t0);
			
		}
		timer_count.Pause();
		printf_timer(timer_count);

		// Compute bundled camera path
		vector<Mat> Vec;
		
		timer_count.Reset();
		timer_count.Start();
		allPath allpath = allPath(cuth, cutw, VecHomo.size()+1);
		Mat homo = Mat::eye(3, 3, CV_32FC1);
		for (int t = 0; t < VecHomo.size(); t++)
		{
			printf("Compute bundled camera path %d \n", t);	
//cerr << VecHomo[t].size() << " = VecHomo[t].size()" << endl;
			for (int i = 0; i < cutw; i++)
				for (int j = 0; j < cuth; j++)
				{
					// Vec9f tmp = VecHomo[t].at<Vec9f>(i, j);
					// allpath.setHomo(i, j, t, tmp);
					allpath.setHomo(i, j, t, VecHomo[t][i][j].inv());
				}

		}
		// allpath.computePath();
		allpath.computePathOnly30Frames();
		timer_count.Pause();
		printf_timer(timer_count);


		timer_count.Reset();
		timer_count.Start();
		allpath.optimizePath(20);
		// allpath.jacobiSolver();
		timer_count.Pause();
		printf_timer(timer_count);

		// allpath.computeWarp();

		vector<Mat> path = allpath.getPath(0, 0);
		vector<Mat> optpath = allpath.getOptimizedPath(0, 0);
		Mat picture(1000, 1000, CV_8UC3, Scalar(255,255,255));  
		vector<Point2f> center(1);
		vector<Point2f> move(1);
		vector<Point2f> stable(1);
		vector<Point2f> tmp(1);
		
		float scale = 4.f;
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
		imwrite("optimize_path.jpg", picture);
		
		//namedWindow("Display window", WINDOW_AUTOSIZE);
		//imshow("Display window", picture );
		//waitKey(0);

		

		// Warp image
		vector<Mat> warp_frames;
		Mat globalH = Mat::eye(3, 3, CV_64FC1);
		
		timer_count.Reset();
		timer_count.Start();
		asapWarp asap = asapWarp(height, width, cuth+1, cutw+1, 1); 
		warp W(asap);
			BundleHomo ppath(cut, Path (cut));
			BundleHomo ipath(cut, Path (cut));
			vector<Point2f> VV(4);
			vector<Point2f> WW(4);
		for (int t = 0; t < min(1000, allpath.time); t++)
		{
			// if (t < 140 || t > 155)
			// 	continue;
			
			printf("Image Synthesis %d \n", t);
			///* my new warpimg method
			Mat warp_frame;

			// get path by cellPoints
			for (int i = 0;i<cut;i++)
				for(int j = 0;j<cut;j++)
				{
					VV[0] = allCellPoints[0][i][j];
					VV[1] = allCellPoints[0][i+1][j];
					VV[2] = allCellPoints[0][i][j+1];
					VV[3] = allCellPoints[0][i+1][j+1];
					WW[0] = allCellPoints[t][i][j];
					WW[1] = allCellPoints[t][i+1][j];
					WW[2] = allCellPoints[t][i][j+1];
					WW[3] = allCellPoints[t][i+1][j+1];
					ppath[i][j] = findHomography(VV, WW);
					ipath[i][j] = Mat::eye(3,3,ppath[i][j].type());
				}
			cout<<"ppath calc done"<<endl;
			// W.warpImageMesh(frames[t], warp_frame, ppath, ipath);
			W.warpImageMesh(frames[t], warp_frame, allpath.getPath(t), allpath.getOptimizedPath(t));
			// W.warpImageMesh(frames[t], warp_frame, allpath.getHomo(t), allpath.getHomo(t));
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

			sprintf(str, "result/frame_warp_%03d.jpg", t);
			imwrite(str, frame_warp);
		}
		timer_count.Pause();
		printf_timer(timer_count);



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


