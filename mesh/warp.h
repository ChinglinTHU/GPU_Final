#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"

#include "asapWarp.h" 

using namespace std;
using namespace cv;
typedef vector<vector<Mat> > BundleHomo;

class warp
{
public:
	warp(asapWarp asap);
	~warp();
	void warpImageMesh(Mat img, Mat & warpimg, BundleHomo C, BundleHomo P);

private:
	Point warpImgByVertex(Mat img, Mat & warpimg, vector<Point2f> pt, vector<Point2f> warppt, 
						bool all, Point offset = Point(0, 0), Size s = Size(100, 100));
	void DrawPoints(Mat img, Mat & pointImg, vector<Point2f> pts, Point offset = Point(0, 0));

	asapWarp asap;
	int width;
	int height;
};