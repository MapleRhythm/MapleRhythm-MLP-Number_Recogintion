#include <iostream>
#include <fstream>
#include <string>
#include<Eigen/Dense>
#include<cmath>
#include<iomanip>
#include"EasyBMP.h"
using namespace std;

#define e 2.7182818
#define LENGTH 784
#define WEIGHTNUM 12544
#define FIRST_NUM 100
#define SECOND_NUM 10
#define ROW 28
#define COL 28
#define PACE 0.5
#define run_times 100
#define one 1

#define CE_shi
#ifdef CE_shi
const int LEBEL_NUM = 500;
const int Ce_shi_num = 500;
#else
const int LEBEL_NUM = 60000;
const int Ce_shi_num = 10000;
#endif
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> WEIGHT;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> DATA;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> BIAS;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> TEMP;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> DIFFERENCE;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> OUTPUT;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> INPUT;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> DESIRE;
typedef Eigen::Vector<double, Eigen::Dynamic> VECTOR;

int judge(OUTPUT& output)
{
	double max = 0;
	int temp = 0;
	for (int i = 0; i < 10; i++)
	{

		if (output(i,0) > max)
		{
			max = output(i,0);
			temp = i;
		}
	}
	return temp;
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
void initialize(WEIGHT& weight1, WEIGHT& weight2)
{
	srand((unsigned)time(NULL));
	for (int i = 0; i < FIRST_NUM; i++)
	{
		for (int j = 0; j < LENGTH ; j++)
			weight1(i, j) = ((double)(rand() % 1000) / 700 - 0.5) * sqrt(1.0 / FIRST_NUM);
	}
	for (int i = 0; i < SECOND_NUM ; i++)
	{
		for (int j = 0; j < FIRST_NUM ; j++)
			weight2(i, j) = ((double)(rand() % 1000) / 700 - 0.5) * sqrt(1.0 / SECOND_NUM );
	}

}

double sigmaori(double a)
{
	return 1.0 / (1 + exp(-a));
}
void ForwardSpread(WEIGHT& weight1, WEIGHT& weight2, INPUT& input, OUTPUT& first_layer, OUTPUT& second_layer)
{
	TEMP z1 = Eigen::Matrix<double, FIRST_NUM, 1 >::Zero(FIRST_NUM, 1);
	TEMP z2 = Eigen::Matrix<double, SECOND_NUM, 1 >::Zero(SECOND_NUM, 1);
	z1 = weight1 * input;
	
	for (int i = 0; i < FIRST_NUM ; i++)
	{
		first_layer(i) = sigmaori(z1(i));
	}
	
	z2 = weight2 * first_layer;
	for (int i = 0; i < SECOND_NUM ; i++)
	{

		second_layer(i) = sigmaori(z2(i));
	}


}
void read_Mnist_Label(string filename, VECTOR& labels)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;

		for (int i = 0; i < LEBEL_NUM; i++)
		{
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			labels(i) = ((double)label);
		}

	}
}
void read_TEST_Label(string filename, VECTOR& labels)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;

		for (int i = 0; i < Ce_shi_num; i++)
		{
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			labels(i) = ((double)label) ;
		}

	}
}

void read_Mnist_Images(string filename, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& images)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		unsigned char label;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);

		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		cout << "rows = " << n_rows << endl;
		cout << "cols = " << n_cols << endl;

		for (int i = 0; i < LEBEL_NUM; i++)
		{
			for (int j = 0; j < n_rows * n_cols; j++)
			{
				unsigned char image = 0;
				file.read((char*)&image, sizeof(image));
				images(i, j) = (double)image/180.0;
			}
		}
	}
}
void read_TEST_Images(string filename, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& images)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		unsigned char label;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);

		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		cout << "rows = " << n_rows << endl;
		cout << "cols = " << n_cols << endl;

		for (int i = 0; i < Ce_shi_num; i++)//记得改回来
		{
			for (int j = 0; j < n_rows * n_cols; j++)
			{
				unsigned char image = 0;
				file.read((char*)&image, sizeof(image));
				images(i, j) = (double)image / 180.0;
			}
		}
	}
}
void ti_du_xia_jiang(int num, INPUT input, WEIGHT& w1, WEIGHT& w2, WEIGHT& dw1, WEIGHT& dw2, OUTPUT& a1, OUTPUT& a2,DIFFERENCE &d1,DIFFERENCE &d2)
{
	int i, j, k;
	TEMP z1 = Eigen::Matrix<double, FIRST_NUM, 1 >::Zero(FIRST_NUM, 1);
	TEMP z2 = Eigen::Matrix<double, SECOND_NUM, 1 >::Zero(SECOND_NUM, 1);
	DESIRE desire = Eigen::Matrix<double, SECOND_NUM,1>::Zero(SECOND_NUM,1);
	desire(num,0) = 1;
	double sum = 0;
	//输出偏差
	/*for (i = 0; i < SECOND_NUM; i++)
	{
		sum += (a2(i) - desire(i)) * (a2(i) - desire(i));
	}
	cout << sum << endl;*/
	

	for (i = 0; i < SECOND_NUM; i++)
	{
		d2(i) = a2(i) - desire(i);
	}
	for(i=0;i<FIRST_NUM;i++)
	{
		z1(i) = a1(i) * (1 - a1(i));
	}
	d1=w2.transpose()* d2;
	for(i=0;i<FIRST_NUM;i++)
	{
		d1(i) = d1(i) * z1(i);
	}

}
void pi_liang_ti_du_xun_lian(VECTOR& labels, DATA& data, WEIGHT& w1, WEIGHT& w2,BIAS &b1,BIAS &b2)
{
	WEIGHT dw1 = Eigen::Matrix<double, FIRST_NUM , LENGTH >::Zero(FIRST_NUM, LENGTH );
	WEIGHT dw2 = Eigen::Matrix<double, SECOND_NUM , FIRST_NUM >::Zero(SECOND_NUM , FIRST_NUM );

	BIAS db1 = Eigen::Matrix<double, FIRST_NUM, 1 >::Zero(FIRST_NUM, 1);
	BIAS db2 = Eigen::Matrix<double, SECOND_NUM, 1 >::Zero(SECOND_NUM, 1);

	OUTPUT a1 = Eigen::Matrix<double, FIRST_NUM ,1>::Zero(FIRST_NUM ,1);
	OUTPUT a2 = Eigen::Matrix<double, SECOND_NUM ,1>::Zero(SECOND_NUM ,1);

	DIFFERENCE d1 = Eigen::Matrix<double, FIRST_NUM, 1>::Zero(FIRST_NUM, 1);
	DIFFERENCE d2 = Eigen::Matrix<double, SECOND_NUM, 1>::Zero(SECOND_NUM, 1);

	INPUT input = Eigen::Matrix<double,1 ,LENGTH>::Zero(1 ,LENGTH);
	




	for (int j = 0; j < run_times; j++)
	{
		
		for (int i = 0; i < LEBEL_NUM; i++)
		{
			input = data.row(i);
			input = input.transpose();
			ForwardSpread(w1, w2, input, a1, a2);
			ti_du_xia_jiang(labels(i), input, w1, w2, dw1, dw2, a1, a2, d1, d2);
			dw1 = dw1 + d1 * input.transpose();
			db1 = db1 + d1;
			dw2 = dw2 + d2 * a1.transpose();
			db2 = db2 + d2;
		}
		dw1 = dw1 / LEBEL_NUM;
		db1 = db1 / LEBEL_NUM;
		dw2 = dw2 / LEBEL_NUM;
		db2 = db2 / LEBEL_NUM;
		w1 -= dw1 * PACE;
		w2 -= dw2 * PACE;
		b1 -= db1 * PACE;
		b2 -= db2 * PACE;
		dw1 = Eigen::Matrix<double, FIRST_NUM, LENGTH >::Zero(FIRST_NUM, LENGTH);
		dw2 = Eigen::Matrix<double, SECOND_NUM, FIRST_NUM >::Zero(SECOND_NUM, FIRST_NUM);
		db1 = Eigen::Matrix<double, FIRST_NUM, 1 >::Zero(FIRST_NUM, 1);
		db2 = Eigen::Matrix<double, SECOND_NUM, 1 >::Zero(SECOND_NUM, 1);
		cout << j << endl;
	}
}

void sui_ji_ti_du_xun_lian(VECTOR& labels, DATA& data, WEIGHT& w1, WEIGHT& w2)
{
	WEIGHT dw1 = Eigen::Matrix<double, FIRST_NUM, LENGTH>::Zero(FIRST_NUM, LENGTH);
	WEIGHT dw2 = Eigen::Matrix<double, SECOND_NUM , FIRST_NUM >::Zero(SECOND_NUM , FIRST_NUM );
	TEMP z1 = Eigen::Vector<double, FIRST_NUM >::Zero(FIRST_NUM);
	TEMP z2 = Eigen::Vector<double, SECOND_NUM >::Zero(SECOND_NUM );
	OUTPUT a1 = Eigen::Vector<double, FIRST_NUM >::Zero(FIRST_NUM );
	OUTPUT a2 = Eigen::Vector<double, SECOND_NUM >::Zero(SECOND_NUM );
	INPUT input = Eigen::Vector<double, LENGTH >::Zero(LENGTH );
	DIFFERENCE d1 = Eigen::Matrix<double, FIRST_NUM, 1>::Zero(FIRST_NUM, 1);
	DIFFERENCE d2 = Eigen::Matrix<double, SECOND_NUM, 1>::Zero(SECOND_NUM, 1);

	z1 = z1.transpose();
	z2 = z2.transpose();
	a1 = a1.transpose();
	a2 = a2.transpose();
	input = input.transpose();
	initialize(w1, w2);
	for (int i = 0; i < LEBEL_NUM; i++)
	{
		input = data.row(i);
		ForwardSpread(w1, w2, input, a1, a2);
		ti_du_xia_jiang(labels(i), input, w1, w2, dw1, dw2 , a1, a2,d1,d2);
			double sum = 0;
		dw1 = dw1 * PACE;
		w1 -= dw1;
		dw2 = dw2 * PACE;
		w2 -= dw2;
	}
}

void xiao_pi_liang_ti_du_xun_lian(VECTOR& labels, DATA& data, WEIGHT& w1, WEIGHT& w2, WEIGHT& w3)
{


	OUTPUT a1 = Eigen::Matrix<double, FIRST_NUM, 1>::Zero(FIRST_NUM, 1);
	OUTPUT a2 = Eigen::Matrix<double, SECOND_NUM, 1>::Zero(SECOND_NUM, 1);

	INPUT input = Eigen::Matrix<double, 1, LENGTH>::Zero(1, LENGTH);
	WEIGHT dw1 = Eigen::Matrix<double, FIRST_NUM, LENGTH >::Zero(FIRST_NUM, LENGTH);
	WEIGHT dw2 = Eigen::Matrix<double, SECOND_NUM, FIRST_NUM >::Zero(SECOND_NUM, FIRST_NUM);
	DIFFERENCE d1 = Eigen::Matrix<double, FIRST_NUM, 1>::Zero(FIRST_NUM, 1);
	DIFFERENCE d2 = Eigen::Matrix<double, SECOND_NUM, 1>::Zero(SECOND_NUM, 1);
	TEMP z1 = Eigen::Matrix<double, FIRST_NUM, 1 >::Zero(FIRST_NUM, 1);
	TEMP z2 = Eigen::Matrix<double, SECOND_NUM, 1 >::Zero(SECOND_NUM, 1);
	z1 = z1.transpose();
	z2 = z2.transpose();

	a1 = a1.transpose();
	a2 = a2.transpose();

	input = input.transpose();

	int num = 30;
	initialize(w1, w2);
	for (int j = 0; j < 50; j++)
	{
		WEIGHT DW1 = Eigen::Matrix<double, FIRST_NUM, LENGTH>::Zero(FIRST_NUM, LENGTH);
		WEIGHT DW2 = Eigen::Matrix<double, SECOND_NUM, FIRST_NUM>::Zero(SECOND_NUM, FIRST_NUM);
		for (int i = 0; i < 6000; i++)
		{
			input = data.row(i);
			ForwardSpread(w1, w2, input, a1, a2);
			ti_du_xia_jiang(labels(i), input, w1, w2, dw1, dw2,a1, a2,d1,d2);
			DW1 += dw1 * 1.0 / num;
			DW2 += dw2 * 1.0 / num;
		}
		w1 -= DW1;
		w2 -= DW2;
		cout << j << endl;
	}
}
	void readV(string filename, VECTOR & input, int length)
{
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open())
	{
		double temp = 0;
		for (int i = 0; i < length; i++)
		{

			file.read((char*)&temp, sizeof(temp));
			input(i) = temp;
		}
	}
	file.close();
}
void writeV(string filename, VECTOR& input, int length)
{
	std::ofstream file(filename, std::ios::binary);
	if (file.is_open())
	{
		for (int i = 0; i < length; i++)
		{

			file.write((char*)&input(i), sizeof(input(i)));
		}
	}
	file.close();
}
void readM(string filename, WEIGHT& input, int row, int col)
{
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open())
	{
		double temp = 0;
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				file.read((char*)&temp, sizeof(temp));
				input(i, j) = temp;
			}
		}
	}
	file.close();
}
void writeM(string filename, WEIGHT& input, int row, int col)
{
	std::ofstream file(filename, std::ios::binary);
	if (file.is_open())
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				file.write((char*)&input(i, j), sizeof(input(i, j)));
			}
		}
	}
	file.close();
}
int main()
{
	clock_t start_time = clock();
	{
		VECTOR labels;
		labels = Eigen::Vector<double, LEBEL_NUM>::Zero(LEBEL_NUM);
		read_Mnist_Label("train-labels.idx1-ubyte", labels);
		DATA data;
		data = Eigen::Matrix<double, LEBEL_NUM, LENGTH>::Zero(LEBEL_NUM, LENGTH);
		read_Mnist_Images("train-images.idx3-ubyte", data);
	   
		WEIGHT w1 = Eigen::Matrix<double, FIRST_NUM , LENGTH >::Zero(FIRST_NUM , LENGTH );
		WEIGHT w2 = Eigen::Matrix<double, SECOND_NUM , FIRST_NUM >::Zero(SECOND_NUM , FIRST_NUM );

		BIAS b1= Eigen::Matrix<double, FIRST_NUM, 1>::Zero(FIRST_NUM, 1);
		BIAS b2 = Eigen::Matrix<double, SECOND_NUM, 1 >::Zero(SECOND_NUM, 1);
		
	
		initialize(w1, w2);
		pi_liang_ti_du_xun_lian(labels, data, w1, w2,b1,b2);

		writeM("w1,dat", w1, FIRST_NUM , LENGTH );
		writeM("w2.dat", w2, SECOND_NUM , FIRST_NUM );
		
	
	}
	clock_t end_time = clock();
	cout << "--------时间--------" << endl;
	cout << "耗时: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << 's' << endl;
	
	return 0;
}