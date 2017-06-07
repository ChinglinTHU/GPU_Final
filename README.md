How to run asap?

under file, run

$ g++ -std=c++11 `pkg-config --cflags --libs ~/local/lib/pkgconfig/opencv.pc` ./mesh/asapWarp.cpp ./path/allPath.cpp main.cpp

$ ./a.out ./data/0.avi

