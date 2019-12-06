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

	// ��ʾ����ת��
	Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
	cout << result << endl;

	// ��������
	cout << "��������" << endl;
	matrix_33 = Eigen::Matrix3d::Random();
	cout << matrix_33 << endl;

	cout << matrix_33.transpose() << endl; // ת��
	cout << matrix_33.sum() << endl;       // ��Ԫ�غ�
	cout << matrix_33.trace() << endl;     // ��
	cout << 10 * matrix_33 << endl;        // ����
	cout << matrix_33.inverse() << endl;   // ��
	cout << matrix_33.determinant() << endl;  // ����ʽ

	cout << "����ֵ��" << endl;
	// ����ֵ
	// ʵ�Գƾ�����Ա�֤�Խǻ��ɹ�
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);

	cout << "Eigen Value : " << eigen_solver.eigenvalues() << endl;
	cout << "Eigen Vector : " << eigen_solver.eigenvectors() << endl;

	// �ⷽ��
	// ��� matrix_NN * x = v_Nd ����
	Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN;
	matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
	Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd;
	v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

	clock_t time_stt = clock(); // ��ʱ
	// ֱ������
	Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
	cout << "time use in normal inverse is "
		 << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
	
	// ����ֽ���⣬���� QR�ֽ⣬
	time_stt = clock();
	x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
	cout << "time use in Qr composition is "
		<< 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
	system("pause");
	return 0;
}


