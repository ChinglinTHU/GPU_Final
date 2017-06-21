#include <string>
#include <iostream>
#include <stdexcept>
#include <vector>

// cv libraries
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videostab.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/opencv_modules.hpp"

/*
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
*/

//#include "opencv2/stitching/detail/blenders.hpp"

#include "./mesh/warp.h"
#include "./mesh/asapWarp.h"
#include "./path/allPath.h"
#include "./utils/Timer.h"
//#include "./feature/MSOP.h"

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
			// if (s > 9 && s < 20)
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
		
		TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
	    Size subPixWinSize(10,10), winSize(31,31);
	    const int MAX_COUNT = 200;
	    bool getfeature = true;
		vector<Point2f> points[2];
		vector<Point2f> now_pts, next_pts;

		printf("Detecting feature: ");
		for (int t = 0; t < gray_frames.size(); t++)
		{
			//printf("Detecting feature: %d \n", t);
			if (getfeature)
			{
				goodFeaturesToTrack(gray_frames[t], now_pts, MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
	        	cornerSubPix(gray_frames[t], now_pts, subPixWinSize, Size(-1,-1), termcrit);
        	}
        	if (t > 0)
	        {
	            vector<uchar> status;
	            vector<float> err;
	            
	            calcOpticalFlowPyrLK(gray_frames[t-1], gray_frames[t], now_pts, next_pts, status, err, winSize,
	                                 3, termcrit, 0, 0.001);

				vector<unsigned char> match_mask;
	   			Mat globalHomo = findHomography(now_pts, next_pts, RANSAC, 4, match_mask);

	   			for (int j = match_mask.size() - 1; j >= 0; j--)
	   				if (match_mask[j] == 0)
	   				{
	   					now_pts.erase(now_pts.begin()+j);
	   					next_pts.erase(next_pts.begin()+j);
	   				}
	   			if (now_pts.size() != next_pts.size())
	   			   	throw runtime_error("matching points have different size\n");
	   			//printf("\t(ransac_num = %d)\n", now_pts.size());
	   			
	   			vec_now_pts.push_back(now_pts);
	   			vec_next_pts.push_back(next_pts);
	   			vec_global_homo.push_back(globalHomo);
   			}
   			if (next_pts.size() < 50)
	   				getfeature = true;
   			swap(next_pts, now_pts);
		}
		timer_count.Pause();
		printf_timer(timer_count);
		
		// model estimation
		int height = frames[0].rows;
		int width = frames[0].cols;
		int cut = 2*2*2;
		int cuth = cut;
		int cutw = cut;
		float quadWidth = width/cutw;
		float quadHeight = height/cuth;
		double weight = 1;
	
		vector<BundleHomo> VecHomo;
		vector<vector<vector<Point2f>>> allCellPoints; // t,i,j
		vector<vector<Point2f>> preCellPoints(cutw+1, vector<Point2f> (cuth+1));
		vector<vector<Point2f>> curCellPoints(cutw+1, vector<Point2f> (cuth+1));
		for(int i = 0; i<cutw+1;i++)
			for(int j = 0;j<cuth+1;j++)
			{
				Point2f p = Point2f(float(i*quadWidth), float(j*quadHeight));
				preCellPoints[i][j] = p;
			}
		allCellPoints.push_back(preCellPoints);

		timer_count.Reset();
		timer_count.Start();

		Timer timer_jacobi, timer_solve;
		
		printf("Computing frame Homographies: ");
		for (int i = 0; i < vec_now_pts.size(); i++)
		// for (int i = 0; i < 10; i++)
		{
			asapWarp asap = asapWarp(height, width, cuth+1, cutw+1, 3); 
			//printf("Computing frame Homographies (%d, %d) \n", i, i+1);	

			asap.SetControlPts(vec_next_pts[i], vec_now_pts[i], vec_global_homo[i].inv());

			timer_solve.Start();
			//asap.Solve();
			timer_solve.Pause();
			// asap.PrintVertex();

			timer_jacobi.Start();
			asap.IterativeSolve(10);
			timer_jacobi.Pause();

//			return 0;

			asap.SolvePoints(preCellPoints, curCellPoints);
			allCellPoints.push_back(curCellPoints);
			preCellPoints = curCellPoints;	
			
		}
		timer_count.Pause();
		printf_timer(timer_count);

		// Compute bundled camera path
		vector<Mat> Vec;
		
		timer_count.Reset();
		timer_count.Start();

		allPath allpath(cuth+1, cutw+1, allCellPoints.size(), allCellPoints);

		// allpath.computePath();
		// allpath.computePathOnly30Frames();

		timer_count.Pause();
		printf_timer(timer_count);


		timer_count.Reset();
		timer_count.Start();
		// allpath.optimizePath(20);
		// allpath.jacobiSolver();
		allpath.jacobiPointSolver();
		timer_count.Pause();
		printf_timer(timer_count);		

		// Warp image
		vector<Mat> warp_frames;
		Mat globalH = Mat::eye(3, 3, CV_64FC1);
		
		timer_count.Reset();
		timer_count.Start();
		asapWarp asap = asapWarp(height, width, cuth+1, cutw+1, 1); 
		warp W(asap);

		/*
		BundleHomo Identity(width-1, vector<Mat> (height-1));
		for (int i = 0; i < width-1; i++)
			for (int j = 0; j < height-1; j++)
			{
				Identity[i][j] = Mat::eye(3, 3, CV_32FC1);
			}
		*/

		printf("Image Synthesis: ");
		for (int t = 0; t < min(1000, allpath.time); t++)
		{
			
			// printf("Image Synthesis %d \n", t);
			///* my new warpimg method
			Mat warp_frame;

			// W.warpImageMesh(frames[t], warp_frame, allpath.getPath(t), Identity);
			// W.warpImageMesh(frames[t], warp_frame, allpath.getPath(t), allpath.getOptimizedPath(t));
			// W.warpImageMeshGPU(frames[t], warp_frame, allpath.getPath(t), allpath.getOptimizedPath(t));
			W.warpImageMeshbyVertexGPU(frames[t], warp_frame, allpath.getcellPoints(t), allpath.getoptPoints(t));
			warp_frames.push_back(warp_frame);

			/*
			printf("Write images %d \n", t);
			
			char str[20];
			Mat frame_warp = Mat::zeros(frames[t].rows + warp_frames[t].rows, frames[t].cols, CV_8UC3);
			frames[t].copyTo(frame_warp(Rect(0, 0, frames[t].cols, frames[t].rows)));
			warp_frames[t].copyTo(frame_warp(Rect(0, frames[t].rows, warp_frames[t].cols, warp_frames[t].rows)));

			sprintf(str, "result/frame_warp_%03d.jpg", t);
			imwrite(str, frame_warp);
			//*/
		}
		timer_count.Pause();
		printf_timer(timer_count);		

		timer_count.Reset();
		timer_count.Start();
		printf("Write images: ");
		for (int t = 0; t < warp_frames.size(); t++)
		{
			// printf("Write images %d \n", t);
			/* write images */
			char str[20];
			Mat frame_warp = Mat::zeros(frames[t].rows + warp_frames[t].rows, frames[t].cols, CV_8UC3);
			frames[t].copyTo(frame_warp(Rect(0, 0, frames[t].cols, frames[t].rows)));
			warp_frames[t].copyTo(frame_warp(Rect(0, frames[t].rows, warp_frames[t].cols, warp_frames[t].rows)));

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