#pragma once

#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/utility.hpp"

#include "opencv2/calib3d.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

typedef vector<Mat> Path;
typedef vector<vector<Mat> > BundleHomo;
typedef Vec<float, 9> Vec9f;
typedef Vec<double, 9> Vec9d;

class allPath
{
public:
	allPath(int height, int width, int t);
	~allPath();
    void setHomo(int i, int j, int t, Mat p);
    void setHomo(int i, int j, int t, Vec9f p);

    Mat getWarpHomo(int i, int j, int t);
    void setPath(int i, int j, Path p);
	Path getPath(int i, int j);
    Mat getPath(int i, int j, int t);
    BundleHomo getPath(int t);
    Path getOptimizedPath(int i, int j);
    BundleHomo getOptimizedPath(int t);
    void computePath();
    void optimizePath(int iter);
    void jacobiSolver(int iter = 20);
    void computeWarp();
    
    int height, width, time; // mesh height,mesh width

    vector<vector<Path> > optPath;
    vector<vector<Path> > tmpPath;
    vector<vector<Path> > cellPath;
    vector<vector<Path> > cellHomo;
    vector<vector<Path> > warpHomo;
private:
    
};