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

	// �ڽǵ� �� ����
	boardSize.width = 9;
	boardSize.height = 6;

	bool flag = 0;
	flag = Uitls::readStringListFromText(inputFileName, imageList);

	if (!imageList.empty() && flag)
		nframe = imageList.size(); // numbers of images
	for (size_t i = 0; i < nframe; i++)
	{
		Mat view, viewGray;
		std::cout << "���ڼ��飺" << i << "��ͼ��" << std::endl;
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
		cvtColor(view, viewGray, COLOR_BGR2GRAY); // ת���Ҷ�ͼ

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

	std::cout << "�궨��ʼ ��" << endl;
	squareSize = 10; // ÿ�����̸�Ĵ�С
	vector<vector<Point3f>> objectPoints; // ����궨���ϵ���ά����
	// �ڲξ���
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	// ÿ��ͼ�еĽǵ�����
	vector<int> point_counts;
	// ����Ļ������
	distCoeffs = Mat::zeros(8, 1, CV_64F);
	// rotation
	vector<Mat> tvecsMat;
	// translation
	vector<Mat> rvecsMat;

	// inital �궨���ϵ���ά����
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

	// ��ʼ��ÿ��ͼ�еĽǵ�����
	for (size_t k = 0; k < nframe; k++) {
		point_counts.push_back(boardSize.width * boardSize.height);
	}

	calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
	std::cout << "���۱궨��� ====== " << std::endl;

	// ����ܺ�
	double total_err = 0.0;

	// ÿ��ͼ���ƽ�����
	double err = 0.0;
	std::vector<Point2f> image_points2; // ���¼���õ���ͶӰ��
	std::cout << "\t ÿ��ͼ��ı궨�� \n";
	for (int i = 0; i < nframe; i++) {
		vector<Point3f> tempPointSet = objectPoints[i];
		/* ͨ���õ�������������Կռ���ά���������ͶӰ���㣬�õ��µ�ͶӰ�� */
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
		/* ������ͶӰ��;�ͶӰ��֮�����z����궨�����Ӧ���Ǳ궨�㷨�ĺû� */
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
		std::cout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
	}

	/* ����ÿ��ͼ�����ת���� */
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	/* ����ڲ� */
	std::cout << "CameraMatrix �� " << endl;
	std::cout << cameraMatrix << std::endl << std::endl;
	/* ����ϵ�� */
	std::cout << "����ϵ����" << endl;
	std::cout << distCoeffs << std::endl << std::endl;
	/* ÿ��ͼ����תƽ�Ʋ��� */
	for (size_t i = 0; i < nframe; i++)
	{
		std::cout << "��" << i + 1 << "��ͼ�����ת���� : " << std::endl;
		std::cout << tvecsMat[i] << std::endl;
		/* ��ת���� ת ��ת���� */
		Rodrigues(tvecsMat[i], rotation_matrix);
		std::cout << "��" << i + 1 << "��ͼ�����ת����" << std::endl;
		std::cout << rotation_matrix << std::endl;
		std::cout << "��" << i + 1 << "��ͼ���ƽ��������" << std::endl;
		std::cout << rvecsMat[i] << std::endl;
	}

	system("pause");
	return 0;
}