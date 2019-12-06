#include <iostream>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

#define MATRIX_SIZE 50 

int main()
{ 

	Eigen::Matrix<float, 2, 3> matrix_23;

	// Vector3d == Matrix<double, 3, 1>
	// Matrix3d == Matrix<double, 3, 3>
	Eigen::Vector3d v_3d;
	Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();

	// 动态大小矩阵
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
	// 等价动态矩阵
	Eigen::MatrixXd matrix_x;

	// 矩阵操作
	// 矩阵输入
	matrix_23 << 1, 2, 3, 4, 5, 6;

	// 矩阵输出
	cout << matrix_23 << endl;

	// 用（）取值
	for (size_t i = 0; i < 1; i++)
	{
		for (size_t j = 0; j < 2; j++)
		{
			cout << matrix_23(i, j) << endl;
		}
	}
	v_3d << 3, 2, 1;

	// 显示类型转换
	Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
	cout << result << endl;

	// 矩阵运算
	cout << "矩阵运算" << endl;
	matrix_33 = Eigen::Matrix3d::Random();
	cout << matrix_33 << endl;

	cout << matrix_33.transpose() << endl; // 转置
	cout << matrix_33.sum() << endl;       // 个元素和
	cout << matrix_33.trace() << endl;     // 迹
	cout << 10 * matrix_33 << endl;        // 数乘
	cout << matrix_33.inverse() << endl;   // 逆
	cout << matrix_33.determinant() << endl;  // 行列式

	cout << "特征值：" << endl;
	// 特征值
	// 实对称矩阵可以保证对角化成功
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);

	cout << "Eigen Value : " << eigen_solver.eigenvalues() << endl;
	cout << "Eigen Vector : " << eigen_solver.eigenvectors() << endl;

	// 解方程
	// 求解 matrix_NN * x = v_Nd 方程
	Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN;
	matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
	Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd;
	v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

	clock_t time_stt = clock(); // 计时
	// 直接求逆
	Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
	cout << "time use in normal inverse is "
		 << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
	
	// 矩阵分解求解，例如 QR分解，
	time_stt = clock();
	x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
	cout << "time use in Qr composition is "
		<< 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
	system("pause");
	return 0;
}


