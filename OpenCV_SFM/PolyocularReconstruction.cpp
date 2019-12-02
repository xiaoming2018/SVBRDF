#include <iostream>
#include <vector>
#include <io.h>
#include <fstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

// ��ʼ�����ƽṹ
void init_structure(Mat k,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<vector<Vec3b>>& colors_for_all,
	vector<vector<DMatch>>& matches_for_all,
	vector<Point3f>& structure,
	vector<vector<int>>& correspond_structure_idx,
	vector<Vec3b>& colors,
	vector<Mat>& rotations,
	vector<Mat>& motions
);

void get_matched_points(
	vector<KeyPoint>& p1,
	vector<KeyPoint>& p2,
	vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2
);

void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
);

bool find_transform(Mat& k, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask);
void maskout_points(vector<Point2f>& p1, Mat& mask);
void maskout_colors(vector<Vec3b>& p1, Mat& mask);
void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure);
void get_objpoints_and_imgpoints(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<Point3f>& structure,
	vector<KeyPoint>& key_points,
	vector<Point3f>& object_points,
	vector<Point2f>& image_points
);
void fusion_structure(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<Point3f>& structure,
	vector<Point3f>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors
);

// ��ȡͼ��������
void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector<vector<Vec3b>>& colors_for_all
);

void match_features(Mat& query, Mat& train, vector<DMatch>& matches);
void match_features(vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all);
void save_structure(string filename, vector<Mat>& rotations, vector<Mat>& motions, vector<Point3f>& structure, vector<Vec3b>& colors);

std::string dir = "/images/";

int main()
{
	vector<string> img_names;
	img_names.push_back(dir + "0000.jpg");

	// �ڲξ���
	Mat K(Matx33d(
		2759.48, 0, 1520.69,
		0, 2764.16, 1006.81,
		0, 0, 1));
	
	// ������
	vector<vector<KeyPoint>> key_points_for_all;
	// ������
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matches_for_all;

	// ��ȡ����ͼ�������
	extract_features(img_names, key_points_for_all, descriptor_for_all, colors_for_all);
	// ������ͼ�����˳�ε�����ƥ��
	match_features(descriptor_for_all, matches_for_all);

	vector<Point3f> structure;
	// �����i��ͼ���е�j�������Ӧ��structure�е�����
	vector<vector<int>> correspond_struct_idx; 
	vector<Vec3b> colors;
	vector<Mat> rotations;
	vector<Mat> motions;

	// ��ʼ���ṹ����ά���ƣ���һ�͵ڶ���
	init_structure(K,
		key_points_for_all,
		colors_for_all,
		matches_for_all,
		structure,
		correspond_struct_idx,
		colors,
		rotations,
		motions);


	system("pause");
	return 0;
}

void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector<vector<Vec3b>>& colors_for_all
	)
{
	key_points_for_all.clear();
	descriptor_for_all.clear();
	Mat image;

	// ��ȡͼ�񣬻�ȡͼ�������㲢����
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);
	for (auto it = image_names.begin(); it != image_names.end(); it++)
	{
		image = imread(*it);
		if (image.empty())
			continue;
		cout << "Extracing freatures : " << *it << endl;

		vector<KeyPoint> key_points;
		Mat descriptor;
		sift->detect(image, key_points);
		sift->compute(image, key_points, descriptor);

		// ��������٣����ų���ͼ��
		if (key_points.size() <= 10)
			continue;

		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);

		// ��ͨ�� �����ͨ����ɫ
		vector<Vec3b> colors(key_points.size());
		for (int i = 0; i < key_points.size(); i++)
		{
			Point2f& p = key_points[i].pt;
			if (p.x <= image.rows && p.y <= image.cols)
				colors[i] = image.at<Vec3b>(p.x, p.y);
		}
		colors_for_all.push_back(colors);
	}
}

void match_features(vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all)
{
	matches_for_all.clear();
	// n ��ͼ������˳���� n-1 ��ƥ��
	for (int i = 0; i < descriptor_for_all.size()-1; i++)
	{
		cout << "Matching images " << i << " - " << i + 1 << endl;
		vector<DMatch> matches;
		match_features(descriptor_for_all[i], descriptor_for_all[i + 1], matches);
		matches_for_all.push_back(matches);
	}
}

void match_features(Mat& query, Mat& train, vector<DMatch>& matches)
{
	vector<vector<DMatch>> knn_matches;
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(query, train, knn_matches, 2);

	// ��ȡ���� Radio Test ����Сƥ��ľ���
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); r++)
	{
		// Radio Test
		if (knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance)
			continue;
		float dist = knn_matches[r][0].distance;
		if (dist < min_dist)
		{
			min_dist = dist;
		}
	}

	matches.clear();
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance > 0.6 * knn_matches[i][1].distance ||
			knn_matches[i][0].distance > 5 * max(min_dist, 10.0f))
			continue;
		// ����ƥ���
		matches.push_back(knn_matches[i][0]);
	}
}

void init_structure(
	Mat K,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<vector<Vec3b>>& colors_for_all,
	vector<vector<DMatch>>& matches_for_all,
	vector<Point3f>& structure,
	vector<vector<int>>& correspond_struct_idx,
	vector<Vec3b>& colors,
	vector<Mat>& rotations,
	vector<Mat>& motions
)
{
	// ����ͷ����ͼ��֮��ı任����
	vector<Point2f> p1, p2;
	vector<Vec3b> c2;
	Mat R, T; // ��ת��ƽ��
	Mat mask; // mask�д�����ĵ����ƥ��㣬������ĵ����ʧ���
	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);
	get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);
	find_transform(K, p1, p2, R, T, mask); // ���Ƿֽ�R T ����

	// ��ͷ����ͼ�������ά�ؽ�
	maskout_points(p1, mask);
	maskout_points(p2, mask);
	maskout_colors(colors, mask);
    
	Mat R0 = Mat::eye(3, 3, CV_64FC1);
	Mat T0 = Mat::zeros(3, 1, CV_64FC1);
	reconstruct(K, R0, T0, R, T, p1, p2, structure);

	// ����任����
	rotations = { R0, R };
	motions = { T0, T };

	// ��correspond_struct_idx�Ĵ�С��ʼ��Ϊ��key_points_for_all��ȫһ��
	correspond_struct_idx.clear();
	correspond_struct_idx.resize(key_points_for_all.size());
	for (size_t i = 0; i < key_points_for_all.size(); i++)
	{
		correspond_struct_idx[i].resize(key_points_for_all[i].size());
	}

	// ��дͷ����ͼ��Ľṹ����
	int idx = 0;
	vector<DMatch>& matches = matches_for_all[0];
	for (size_t i = 0; i < matches.size(); i++)
	{
		if (mask.at<uchar>(i) == 0)
			continue;
		correspond_struct_idx[0][matches[i].queryIdx] = idx;
		correspond_struct_idx[1][matches[i].trainIdx] = idx;
		++idx;
	}
}