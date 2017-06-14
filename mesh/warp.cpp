#include <string>
#include <iostream>
#include <stdexcept>
#include <vector>

// #include <opencv2/opencv.hpp>
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

#include "warp.h"

using namespace std;
using namespace cv;
using namespace cv::detail;

typedef Vec<float, 9> Vec9f;
typedef Vec<double, 9> Vec9d;
typedef vector<vector<Mat> > BundleHomo;
#define index(i, j) (j)*width+(i)


warp::warp(asapWarp asap)
{
	this->asap = asap;
}

warp::~warp(){}


void warp::warpImageMesh(Mat img, Mat & warpimg, BundleHomo C, BundleHomo P)
{

	this->height = asap.height;
	this->width = asap.width;
	vector<Point2f> cellPtsT(width*height);
	vector<Point2f> cellPts0(width*height);
	
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{
			cellPtsT[index(i, j)] = asap.compute_pos(i, j);
		}

	vector<Point2f> pt(1);
	vector<Point2f> warpPt(1);

	float minx, miny, maxx, maxy;
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{
			int N = 0;
			Point2f sumpt(0.f, 0.f);
			pt[0] = cellPtsT[index(i, j)];
//cerr << "(" << i << ", " << j << ")" << endl;
			for (int I = max(0, i-1); I < min(width-1, i+1); I++)
				for (int J = max(0, j-1); J < min(height-1, j+1); J++)
				{
					N++;
					perspectiveTransform(pt, warpPt, C[I][J]);
					// perspectiveTransform(pt, warpPt, C[I][J].inv());
//cerr << "\t(" << I << ", " << J << ") " << endl;
//cerr << "C[I][J].inv() = " << C[I][J].inv() << endl;
//cerr << "warpPt = " << warpPt[0] << endl;
					sumpt += warpPt[0];
				}

			cellPts0[index(i, j)] = sumpt / N;
//cerr << "(" << i << ", " << j << ") = " << cellPts0[index(i, j)] << endl;
			if (i == 0 && j == 0)
			{
				minx = cellPts0[index(i, j)].x;
				maxx = cellPts0[index(i, j)].x;
				miny = cellPts0[index(i, j)].y;
				maxy = cellPts0[index(i, j)].y;
			}
			else
			{
				minx = min(cellPts0[index(i, j)].x, minx);
				maxx = max(cellPts0[index(i, j)].x, maxx);
				miny = min(cellPts0[index(i, j)].y, miny);
				maxy = max(cellPts0[index(i, j)].y, maxy);
			}

 		}

	float dist = max(maxx-minx, maxy-miny); 
	float rate = 1.f;
cerr << "dist = " << dist << endl;
	if (dist > 4000.f)
	{
		rate = 4000.f/dist;
		for (int i = 0; i < cellPts0.size(); i++)
			cellPts0[i] = cellPts0[i] * rate;
	}

	Mat originImg;
	Point2f offset = warpImgByVertex(img, originImg, cellPtsT, cellPts0, true);
	/* imshow
	namedWindow("originImg", WINDOW_AUTOSIZE);
	imshow("originImg", originImg);
	waitKey(0);
	//*/

	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{
			int N = 0;
			Point2f sumpt(0.f, 0.f);
			pt[0] = cellPtsT[index(i, j)];
			for (int I = max(0, i-1); I < min(width-1, i+1); I++)
				for (int J = max(0, j-1); J < min(height-1, j+1); J++)
				{
					N++;
					perspectiveTransform(pt, warpPt, P[I][J]);
					// perspectiveTransform(pt, warpPt, P[I][J].inv());
					sumpt += warpPt[0];
				}

			cellPts0[index(i, j)] = sumpt / N;
////cerr << "(" << i << ", " << j << ") = " << cellPts0[index(i, j)] << endl;			
		}

	if (dist > 4000.f)
	{
		rate = 4000.f/dist;
		for (int i = 0; i < cellPts0.size(); i++)
			cellPts0[i] = cellPts0[i] * rate;
	}

	warpImgByVertex(originImg, warpimg, cellPts0, cellPtsT, false, offset, img.size());
	originImg.release();

	/* imshow
	namedWindow("warpimg", WINDOW_AUTOSIZE);
	imshow("warpimg", warpimg);
	waitKey(0);
	//*/

}

Point warp::warpImgByVertex(Mat img, Mat & warpimg, vector<Point2f> pt, vector<Point2f> warppt, 
						bool all, Point offset, Size s)
{
	// offset means that offset -> (0, 0) 

	int minx = img.size().width;
	int maxx = 0;
	int miny = img.size().height;
	int maxy = 0;
	for (int i = 0; i < pt.size(); i++)
	{
		pt[i].x += offset.x;
		pt[i].y += offset.y;
		warppt[i].x += offset.x;
		warppt[i].y += offset.y;

		minx = min(int(floor(warppt[i].x)), minx);
		miny = min(int(floor(warppt[i].y)), miny);
		maxx = max(int(ceil(warppt[i].x)), maxx);
		maxy = max(int(ceil(warppt[i].y)), maxy);
	}
	offset.x -= minx;
	offset.y -= miny;

	for (int i = 0; i < pt.size(); i++)
	{
		warppt[i].x -= minx;
		warppt[i].y -= miny;
	}
	maxx -= minx;
	minx -= minx;
	maxy -= miny;
	miny -= miny;
	
	int sizex = maxx - minx + 1;
	int sizey = maxy - miny + 1;

	FeatherBlender blender(0.5f);  //sharpness
	blender.prepare(Rect(0, 0, sizex, sizey));

	Mat frame;
	img.convertTo(frame, CV_16SC3);

	for (int i = 0; i < width-1; i++)
		for (int j = 0; j < height-1; j++)
		{
			Mat warp_frame, mask, h;
			mask = Mat::zeros(Size(sizex, sizey), CV_8U);

			Point countour[1][4];
			countour[0][0] = Point(warppt[index(i, j)]);
			countour[0][1] = Point(warppt[index(i, j+1)]);
			countour[0][2] = Point(warppt[index(i+1, j+1)]);
			countour[0][3] = Point(warppt[index(i+1, j)]);			

			const Point* ppt[1] = { countour[0] }; 
			int npt[] = {4};  
			fillPoly(mask, ppt, npt, 1, Scalar::all(255)); 
			polylines(mask, ppt, npt, 1, 1, Scalar::all(255), 10);  

			vector<Point2f> P, WP;
			for (int I = i; I < i+2; I++)
				for (int J = j; J < j+2; J++)
				{
					P.push_back(pt[index(I, J)]);
					WP.push_back(warppt[index(I, J)]);
				}

			h = findHomography(P, WP);
////cerr << "(" << i << ", " << j << ") = " << h << endl;	
			warpPerspective(frame, warp_frame, h, mask.size());
			// fb->feed(warp_frame, mask, Point(0, 0));
			blender.feed(warp_frame, mask, Point(0, 0));
		}

	Mat mask = Mat::zeros(Size(sizex, sizey), CV_8U);
	if (!all)
	{
		minx = offset.x;
		maxx = min(offset.x + s.width - 1, s.width- 1);
		miny = offset.y;
		maxy = min(offset.y + s.height - 1, s.height - 1);
	}
	else
	{
		s.width = sizex;
		s.height = sizey;
	}
	sizex = maxx - minx + 1;
	sizey = maxy - miny + 1;

	Mat warp_frame;

	mask(Rect(minx, miny, sizex, sizey)).setTo(Scalar::all(255));
	// fb->blend(warp_frame, mask);
	blender.blend(warp_frame, mask);
	//free(fb);
	warp_frame.convertTo(warp_frame, CV_8UC3);

	/* Draw Points on warpframe
	Mat warp_frame_points;
	DrawPoints(warp_frame, warp_frame_points, warppt, Point(0, 0));
	namedWindow("warp_frame_points", WINDOW_NORMAL);
	imshow("warp_frame_points", warp_frame_points);
	waitKey(0);
	//*/

	warpimg = Mat::zeros(s, warp_frame.type());
	warp_frame(Rect(minx, miny, sizex, sizey)).copyTo(warpimg(Rect(0, 0, sizex, sizey)));

	frame.release();
	warp_frame.release();
	mask.release();

	return offset;
}

void warp::DrawPoints(Mat img, Mat & pointImg, vector<Point2f> pts, Point offset)
{
	pointImg = img.clone();
	for (int i = 0; i < pts.size(); i++)
	{
		circle(pointImg, Point(pts[i])+offset, 3, Scalar(0, 0, 255), -1);
	}
}

#undef index
