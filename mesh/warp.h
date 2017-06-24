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
	void warpImageMeshGPU(Mat img, Mat & warpimg, BundleHomo C, BundleHomo P);
	void warpImageMeshbyVertexGPU(Mat img, Mat & warpimg, vector<Point2f> warpPts0, vector<Point2f> warpPtsT, int* cutxy);

private:
	Point warpImgByVertex(Mat img, Mat & warpimg, vector<Point2f> pt, vector<Point2f> warppt, 
						bool all, Point offset = Point(0, 0), Size s = Size(100, 100));
	void DrawPoints(Mat img, Mat & pointImg, vector<Point2f> pts, Point offset = Point(0, 0),  Scalar color = Scalar(0, 0, 255));
	void compute_homo(float *C, const vector<Point2f> &pts, const vector<Point2f> &warpPts);
	void findCut(Mat img, int* cutxy, vector<Point2i> corner);
	vector<Point2i> compute_corner(vector<Point2f> warpPts0, float *Pinv);

	asapWarp asap;
	int width;
	int height;
	int cellwidth;
	int cellheight;
	vector<Point2f> cellPtsT; 
	vector<Point2f> cellPts0;
};