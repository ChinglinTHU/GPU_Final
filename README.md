How to run asap?

under file, run

$ nvcc -std=c++11 -lcusparse -lcusolver -lcudart -lcusparse -lcuda -Xcompiler -fopenmp `pkg-config --cflags --libs ~/local/lib/pkgconfig/opencv.pc` utils/cuSpSolver.cu mesh/asapWarp.cpp ./mesh/warp.cu path/allPath.cpp main.cpp && ./a.out data/0.avi

$ ./a.out ./data/0.avi

