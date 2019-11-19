#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "utils.h"

using namespace cv;

int main()
{
	Size boardSize, imageSize;
	float squareSize, aspectRatio;
	Mat cameraMatrix, distCoeffs;
	string inputFileName = "source/text.txt";
	int nframe = 0; // pic number

	vector<vector<Point2f>> imagePoints;
	vector<string> imageList;

	bool flag = 0;
	//flag = Uitls::readStringListFromYml(inputFileName, imageList);
	flag = Uitls::readStringListFromText(inputFileName, imageList);

	if (!imageList.empty() && flag)
		nframe = imageList.size();


	system("pause");
	return 0;
}