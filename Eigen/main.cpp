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

	// ��̬��С����
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
	// �ȼ۶�̬����
	Eigen::MatrixXd matrix_x;

	// �������
	// ��������
	matrix_23 << 1, 2, 3, 4, 5, 6;

	// �������
	cout << matrix_23 << endl;
	
	// �ã���ȡֵ
	for (size_t i = 0; i < 1; i++)
	{
		for (size_t j = 0; j < 2; j++)
		{
			cout << matrix_23(i, j) << endl;
		}
	}
	v_3d << 3, 2, 1;
	Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;

	cout << result << endl;


	system("pause");
	return 0;
}


