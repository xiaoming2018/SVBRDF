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
	Mat distCoeffs;
	string inputFileName = "source/text.txt";
	int nframe = 0; // pic number

	vector<vector<Point2f>> imagePoints;
	vector<string> imageList;

	// 内角点 行 列数
	boardSize.width = 9;
	boardSize.height = 6;

	bool flag = 0;
	flag = Uitls::readStringListFromText(inputFileName, imageList);

	if (!imageList.empty() && flag)
		nframe = imageList.size(); // numbers of images
	for (size_t i = 0; i < nframe; i++)
	{
		Mat view, viewGray;
		std::cout << "正在检验：" << i << "副图像" << std::endl;
		if (i < (int)imageList.size())
			view = imread(imageList[i], 1);

		if (i == 0) {
			imageSize.width = view.cols;
			imageSize.height = view.rows;
			std::cout << "imageSize.width : " << imageSize.width << std::endl;
			std::cout << "imageSize.height : " << imageSize.height << std::endl;
		}

		/*  */
		vector<Point2f> pointbuf;
		cvtColor(view, viewGray, COLOR_BGR2GRAY); // 转换灰度图

		// find corners
		bool found = findChessboardCorners(view, boardSize, pointbuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

		// improve the found corners' coordiante accuracy
		if (found) {
			cornerSubPix(viewGray, pointbuf, Size(11, 11), Size(-1, -1),
				TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
			// storge
			imagePoints.push_back(pointbuf);
			// paint corners
			drawChessboardCorners(view, boardSize, Mat(pointbuf), found);
		}
	}

	std::cout << "标定开始 ：" << endl;
	squareSize = 10; // 每个棋盘格的大小
	vector<vector<Point3f>> objectPoints; // 保存标定板上的三维坐标
	// 内参矩阵
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	// 每幅图中的角点数量
	vector<int> point_counts;
	// 相机的畸变参数
	distCoeffs = Mat::zeros(8, 1, CV_64F);
	// rotation
	vector<Mat> tvecsMat;
	// translation
	vector<Mat> rvecsMat;

	// inital 标定板上的三维坐标
	for (int t = 0; t < nframe; t++)
	{
		vector<Point3f> tempPointSet;
		for (int i = 0; i < boardSize.height; i++)
		{
			for (int j = 0; j < boardSize.width; j++)
			{
				Point3f realPoint;
				realPoint.x = i * squareSize;
				realPoint.y = j * squareSize;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		objectPoints.push_back(tempPointSet);
	}

	// 初始化每幅图中的角点数量
	for (size_t k = 0; k < nframe; k++) {
		point_counts.push_back(boardSize.width * boardSize.height);
	}

	calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
	std::cout << "评价标定结果 ====== " << std::endl;

	// 误差总和
	double total_err = 0.0;

	// 每幅图像的平均误差
	double err = 0.0;
	std::vector<Point2f> image_points2; // 重新计算得到的投影点
	std::cout << "\t 每幅图像的标定误差： \n";
	for (int i = 0; i < nframe; i++) {
		vector<Point3f> tempPointSet = objectPoints[i];
		/* 通过得到的相机参数，对空间三维点进行重新投影计算，得到新的投影点 */
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
		/* 计算新投影点和旧投影点之间的误差，z这个标定结果反应的是标定算法的好坏 */
		vector<Point2f> tempImagePoint = imagePoints[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (size_t j = 0; j < tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		std::cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}

	/* 保存每幅图像的旋转矩阵 */
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	/* 相机内参 */
	std::cout << "CameraMatrix ： " << endl;
	std::cout << cameraMatrix << std::endl << std::endl;
	/* 畸变系数 */
	std::cout << "畸变系数：" << endl;
	std::cout << distCoeffs << std::endl << std::endl;
	/* 每幅图的旋转平移参数 */
	for (size_t i = 0; i < nframe; i++)
	{
		std::cout << "第" << i + 1 << "副图像的旋转向量 : " << std::endl;
		std::cout << tvecsMat[i] << std::endl;
		/* 旋转向量 转 旋转矩阵 */
		Rodrigues(tvecsMat[i], rotation_matrix);
		std::cout << "第" << i + 1 << "幅图像的旋转矩阵：" << std::endl;
		std::cout << rotation_matrix << std::endl;
		std::cout << "第" << i + 1 << "幅图像的平移向量：" << std::endl;
		std::cout << rvecsMat[i] << std::endl;
	}

	system("pause");
	return 0;
}