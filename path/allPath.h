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

    void setPath(int i, int j, Path p);
    Mat getPath(int i, int j, int t);
    Mat getWarpHomo(int i, int j, int t);
    Path getPath(int i, int j);
    Path getOptimizedPath(int i, int j);
    BundleHomo getPath(int t);
    BundleHomo getHomo(int t);
    BundleHomo getOptimizedPath(int t);

    void computePath40FramesWithWeight();
    void computePathOnly30Frames();
    void computePath();
    void optimizePath(int iter);
    void jacobiSolver(int iter = 20);
    void computeWarp();

    /* Panda's Code
    void computeBPath();
    vector<Path> getcellPath(int t);
    vector<Path> gethomoPath(int t);
    vector<Path> getoptPath(int t);
    vector<Path> getbPath(int t);
    //*/ 

    
    int height, width, time; // mesh height,mesh width

    vector<vector<Path> > optPath;
    vector<vector<Path> > tmpPath;
    vector<vector<Path> > cellPath;
    vector<vector<Path> > cellHomo;
    vector<vector<Path> > warpHomo;

private:
    /* Panda's Code
    vector<vector<Path> > BPath;
    */ 
};