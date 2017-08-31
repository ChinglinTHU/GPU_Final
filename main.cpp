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

void SmoothCut(vector<int> & vec_minx, vector<int> & vec_maxx, vector<int> & vec_miny, vector<int> & vec_maxy);
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
			if (true)
			//if (s < 400)
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
		for(int i = 0; i < cutw+1; i++)
			for(int j = 0; j < cuth+1; j++)
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

			timer_jacobi.Start();
			asap.IterativeSolve(20);
			timer_jacobi.Pause();
			//asap.PrintVertex();

			//return 0;

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
		/*
		BundleHomo Identity(width-1, vector<Mat> (height-1));
		for (int i = 0; i < width-1; i++)
			for (int j = 0; j < height-1; j++)
			{
				Identity[i][j] = Mat::eye(3, 3, CV_32FC1);
			}
		*/


		timer_count.Reset();
		timer_count.Start();
		asapWarp asap = asapWarp(height, width, cuth+1, cutw+1, 1); 
		warp W(asap);
		int cutxy[4] = {0};
		int nowcutxy[4];
		printf("Image Synthesis: ");
		vector<int> vec_minx, vec_maxx, vec_miny, vec_maxy;
		for (int t = 0; t < allpath.time; t++)
		{
			//printf("Image Synthesis %d \n", t);

			///* my new warpimg method
			Mat warp_frame;

			// W.warpImageMesh(frames[t], warp_frame, allpath.getPath(t), Identity);
			// W.warpImageMesh(frames[t], warp_frame, allpath.getPath(t), allpath.getOptimizedPath(t));
			// W.warpImageMeshGPU(frames[t], warp_frame, allpath.getPath(t), allpath.getOptimizedPath(t));
			W.warpImageMeshbyVertexGPU(frames[t], warp_frame, allpath.getcellPoints(t), allpath.getoptPoints(t), nowcutxy);
			warp_frames.push_back(warp_frame);
			/*
			cout << "time: " << t << endl;
			cout << "\tminx:  " << nowcutxy[0] << "\tmaxx: " << nowcutxy[1] << endl;
			cout << "\tminy:  " << nowcutxy[2] << "\tmaxy: " << nowcutxy[3] << endl;
			//*/

			/*
			cout << "cellPts: " << endl;
			vector<Point2f> cellpts = allpath.getcellPoints(t);
			for (int j = 0; j < cuth+1; j++)
			{
				for (int i = 0; i < cutw+1; i++)
					cout << cellpts[j*(cutw+1)+i] << ", ";
				cout << endl;
			}
			cout << "optPts: " << endl;
			cellpts = allpath.getoptPoints(t);
			for (int j = 0; j < cuth+1; j++)
			{
				for (int i = 0; i < cutw+1; i++)
					cout << cellpts[j*(cutw+1)+i] << ", ";
				cout << endl;
			}
			//*/


			// minx, maxx, miny, maxy
			vec_minx.push_back(nowcutxy[0]);
			vec_maxx.push_back(nowcutxy[1]);
			vec_miny.push_back(nowcutxy[2]);
			vec_maxy.push_back(nowcutxy[3]);
			if (t == 0)
			{
				cutxy[0] = nowcutxy[0];
				cutxy[1] = nowcutxy[1];
				cutxy[2] = nowcutxy[2];
				cutxy[3] = nowcutxy[3];
			}
			else
			{
				if (cutxy[0] < nowcutxy[0])
					cutxy[0] = nowcutxy[0];
				if (cutxy[1] > nowcutxy[1])
					cutxy[1] = nowcutxy[1];
				if (cutxy[2] < nowcutxy[2])
					cutxy[2] = nowcutxy[2];
				if (cutxy[3] > nowcutxy[3])
					cutxy[3] = nowcutxy[3];
			}


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

		//SmoothCut(vec_minx, vec_maxx, vec_miny, vec_maxy);

		///*
		int minx = cutxy[0];
		int maxx = cutxy[1];
		int miny = cutxy[2];
		int maxy = cutxy[3];
		cout << "origin cut: " << "x - (" << minx << ", " << maxx << "), y - (" << miny << ", " << maxy << ")" << endl;
		int sizex = maxx-minx+1;
		int sizey = maxy-miny+1;
		float cutw_h = float(sizex) / float(sizey);
		float w_h = float(frames[0].cols) / float(frames[0].rows);

		if (cutw_h < w_h)
		{
			int truey = floor(sizex / w_h);
			miny -= (truey - sizey)/2;
			maxy += (truey - sizey)/2;
			miny = miny < 0 ? 0 : miny;
			maxy = maxy > frames[0].rows-1 ? frames[0].rows-1 : maxy;
		}
		else
		{
			int truex = floor(w_h*sizey);
			minx -= (truex - sizex)/2;
			maxx += (truex - sizex)/2;
			minx = minx < 0 ? 0 : minx;
			maxx = maxx > frames[0].cols-1 ? frames[0].cols-1 : maxx;
		}

		/* don't cut
		cout << "don't cut" << endl;
		cout << "origin cut: " << "x - (" << minx << ", " << maxx << "), y - (" << miny << ", " << maxy << ")" << endl;
		minx = miny = 0;
		maxx = frames[0].cols-1;
		maxy = frames[0].rows-1;
		//*/

		sizex = maxx-minx+1;
		sizey = maxy-miny+1;

		cout << "Resize rate: " << float (sizex*sizey) / float(frames[0].rows*frames[0].cols) << endl;
		//*/

		vector<Mat> cut_frames;
		timer_count.Reset();
		timer_count.Start();
		float rate = 0.f;
		printf("Cut images: ");
		for (int t = 0; t < warp_frames.size(); t++)
		{
			// printf("Write images %d \n", t);
			/* cut image and resize */
			/*

			int minx = vec_minx[t];
			int maxx = vec_maxx[t];
			int miny = vec_miny[t];
			int maxy = vec_maxy[t];
			int sizex = maxx-minx+1;
			int sizey = maxy-miny+1;
			float cutw_h = float(sizex) / float(sizey);
			float w_h = float(frames[t].cols) / float(frames[t].rows);

			if (cutw_h < w_h)
			{
				int truey = floor(sizex / w_h);
				miny -= (truey - sizey)/2;
				maxy += (truey - sizey)/2;
				miny = miny < 0 ? 0 : miny;
				maxy = maxy > frames[t].rows-1 ? frames[t].rows-1 : maxy;
			}
			else
			{
				int truex = floor(w_h*sizey);
				minx -= (truex - sizex)/2;
				maxx += (truex - sizex)/2;
				minx = minx < 0 ? 0 : minx;
				maxx = maxx > frames[t].cols-1 ? frames[t].cols-1 : maxx;
			}

			sizex = maxx-minx+1;
			sizey = maxy-miny+1;
			if (t == 0)
				rate = float (sizex*sizey) / float(frames[t].rows*frames[t].cols);
			else if (rate > float (sizex*sizey) / float(frames[t].rows*frames[t].cols))
				rate = float (sizex*sizey) / float(frames[t].rows*frames[t].cols);

			cout << "Resize rate " << t << ": " << float (sizex*sizey) / float(frames[t].rows*frames[t].cols) << endl;
			//*/
			Mat cutimg;
			resize(warp_frames[t](Rect(minx, miny, sizex, sizey)), cutimg, frames[t].size());
			cut_frames.push_back(cutimg);
		}
		timer_count.Pause();
		printf_timer(timer_count);

		/*
		timer_count.Reset();
		timer_count.Start();
		printf("Write images: ");
		for (int t = 0; t < cut_frames.size(); t++)
		{
			// write images 
			char str[20];
			Mat frame_warp = Mat::zeros(frames[t].rows + cut_frames[t].rows, frames[t].cols, CV_8UC3);
			frames[t].copyTo(frame_warp(Rect(0, 0, frames[t].cols, frames[t].rows)));
			cut_frames[t].copyTo(frame_warp(Rect(0, frames[t].rows, cut_frames[t].cols, cut_frames[t].rows)));

			sprintf(str, "result/frame_warp_%03d.jpg", t);
			imwrite(str, frame_warp);
		}
		//cout << "min cut rate = " << rate << endl;
		timer_count.Pause();
		printf_timer(timer_count);
		//*/


		string::size_type pAt = inputPath.find_last_of('.');
		const string NAME = inputPath.substr(0, pAt) + "_stab.avi";
		cout << "write video to " << NAME << endl;
		VideoWriter outputVideo;  
		outputVideo.open(NAME, VideoWriter::fourcc('X','V','I','D'), outputFps, cut_frames[0].size(), true);
		if (!outputVideo.isOpened())
	    {
	        cout  << "Could not open the output video for write: " << NAME << endl;
	        return -1;
	    }

	    ///*

		timer_count.Reset();
		timer_count.Start();
		printf("Write video: "); 
		for (int t = 0; t < cut_frames.size(); t++)
		{
			outputVideo.write(cut_frames[t]);
		}
		timer_count.Pause();
		printf_timer(timer_count); 
		//*/
        namedWindow("Video Player", WINDOW_AUTOSIZE);
        for (int t = 0; t < cut_frames.size(); t++)
        {   
            
            Mat frame_warp = Mat::zeros(frames[t].rows + cut_frames[t].rows, frames[t].cols, CV_8UC3);
            frames[t].copyTo(frame_warp(Rect(0, 0, frames[t].cols, frames[t].rows)));
            cut_frames[t].copyTo(frame_warp(Rect(0, frames[t].rows, cut_frames[t].cols, cut_frames[t].rows)));

            //imshow("Video Player", cut_frames[t]);
            imshow("Video Player", frame_warp);
            char c = waitKey(2);
            if(t == cut_frames.size() - 1)
                t = 0;
            //按ESC退出
            if (c == 27) break;
        }
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

void SmoothCut(vector<int> & minx, vector<int> & maxx, vector<int> & miny, vector<int> & maxy)
{

	vector<int> modifyminx, modifyminy, modifymaxx, modifymaxy;
	modifyminx = minx;

	int n;

	for (int k = 0; k < 10; k++)
	{
		for (int i = 0; i < int(modifyminx.size())-1; i++)
		{
		    if (abs(modifyminx[i]-modifyminx[i+1]) > 1)
		    {	
		        n = abs(modifyminx[i]-modifyminx[i+1]);
		        if (modifyminx[i+1] > modifyminx[i])
		        {
		            for (int j = i+1; j >= max(i+1-n, 0); j--)
		            {
		                modifyminx[j] = modifyminx[i+1]-(i+1)+j;
		                if (modifyminx[j] < minx[j])
		                    modifyminx[j] = minx[j];
		            }
		        }
		        else
		        {
		            for (int j = i; j <= min(i+n, int(modifyminx.size())-1); j++)
		            {
		                modifyminx[j] = modifyminx[i]+i-j;
		                if (modifyminx[j] < minx[j])
		                    modifyminx[j] = minx[j];
		            }
		        }
		    }
		}
		minx = modifyminx;
	}

	modifyminy = miny;

	for (int k = 0; k < 10; k++)
	{
		for (int i = 0; i < int(modifyminy.size())-1; i++)
		{
		    if (abs(modifyminy[i]-modifyminy[i+1]) > 1)
		    {	
		        n = abs(modifyminy[i]-modifyminy[i+1]);
		        if (modifyminy[i+1] > modifyminy[i])
		        {
		            for (int j = i+1; j >= max(i+1-n, 0); j--)
		            {
		                modifyminy[j] = modifyminy[i+1]-(i+1)+j;
		                if (modifyminy[j] < miny[j])
		                    modifyminy[j] = miny[j];
		            }
		        }
		        else
		        {
		            for (int j = i; j <= min(i+n, int(modifyminy.size())-1); j++)
		            {
		                modifyminy[j] = modifyminy[i]+i-j;
		                if (modifyminy[j] < miny[j])
		                    modifyminy[j] = miny[j];
		            }
		        }
		    }
		}
		miny = modifyminy;
	}

	modifymaxx = maxx;

	for (int k = 0; k < 10; k++)
	{
		for (int i = 0; i < int(modifymaxx.size())-1; i++)
		{
		    if (abs(modifymaxx[i]-modifymaxx[i+1]) > 1)
		    {	
		        n = abs(modifymaxx[i]-modifymaxx[i+1]);
		        if (modifymaxx[i+1] < modifymaxx[i])
		        {
		            for (int j = i+1; j >= max(i+1-n, 0); j--)
		            {
		                modifymaxx[j] = modifymaxx[i+1]+(i+1)-j;
		                if (modifymaxx[j] > maxx[j])
		                    modifymaxx[j] = maxx[j];
		            }
		        }
		        else
		        {
		            for (int j = i; j <= min(i+n, int(modifymaxx.size())-1); j++)
		            {
		                modifymaxx[j] = modifymaxx[i]+j-(i);
		                if (modifymaxx[j] > maxx[j])
		                    modifymaxx[j] = maxx[j];
		            }
		        }
		    }
		}
		maxx = modifymaxx;
	}

	modifymaxy = maxy;

	for (int k = 0; k < 10; k++)
	{
		for (int i = 0; i < int(modifymaxy.size())-1; i++)
		{
		    if (abs(modifymaxy[i]-modifymaxy[i+1]) > 1)
		    {	
		        n = abs(modifymaxy[i]-modifymaxy[i+1]);
		        if (modifymaxy[i+1] < modifymaxy[i])
		        {
		            for (int j = i+1; j >= max(i+1-n, 0); j--)
		            {
		                modifymaxy[j] = modifymaxy[i+1]+(i+1)-j;
		                if (modifymaxy[j] > maxy[j])
		                    modifymaxy[j] = maxy[j];
		            }
		        }
		        else
		        {
		            for (int j = i; j <= min(i+n, int(modifymaxy.size())-1); j++)
		            {
		                modifymaxy[j] = modifymaxy[i]+j-(i);
		                if (modifymaxy[j] > maxy[j])
		                    modifymaxy[j] = maxy[j];
		            }
		        }
		    }
		}
		maxy = modifymaxy;
	}

}