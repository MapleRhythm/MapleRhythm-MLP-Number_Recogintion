#include <iostream>
#include <fstream>
#include <string>
#include<Eigen/Dense>
#include<cmath>
#include<iomanip>

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

		if (output(i, 0) > max)
		{
			max = output(i, 0);
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
double sigmaori(double a)
{
	return 1.0 / (1 + exp(-a));
}
void ForwardSpread(WEIGHT& weight1, WEIGHT& weight2, INPUT& input, OUTPUT& first_layer, OUTPUT& second_layer)
{
	TEMP z1 = Eigen::Matrix<double, FIRST_NUM, 1 >::Zero(FIRST_NUM, 1);
	TEMP z2 = Eigen::Matrix<double, SECOND_NUM, 1 >::Zero(SECOND_NUM, 1);
	z1 = weight1 * input;

	for (int i = 0; i < FIRST_NUM; i++)
	{
		first_layer(i) = sigmaori(z1(i));
	}

	z2 = weight2 * first_layer;
	for (int i = 0; i < SECOND_NUM; i++)
	{

		second_layer(i) = sigmaori(z2(i));
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
			labels(i) = ((double)label);
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
void readV(string filename, VECTOR& input, int length)
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
int main()
{
	
		WEIGHT w1 = Eigen::Matrix<double, FIRST_NUM, LENGTH >::Zero(FIRST_NUM, LENGTH);
		WEIGHT w2 = Eigen::Matrix<double, SECOND_NUM, FIRST_NUM >::Zero(SECOND_NUM, FIRST_NUM);

		BIAS b1 = Eigen::Matrix<double, FIRST_NUM, 1>::Zero(FIRST_NUM, 1);
		BIAS b2 = Eigen::Matrix<double, SECOND_NUM, 1 >::Zero(SECOND_NUM, 1);

		readM("w1,dat", w1, FIRST_NUM, LENGTH);
		readM("w2.dat", w2, SECOND_NUM, FIRST_NUM);

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> data2;
		data2 = Eigen::Matrix<double, Ce_shi_num, LENGTH>::Zero(Ce_shi_num, LENGTH);
		read_TEST_Images("t10k-images.idx3-ubyte", data2);


		VECTOR labels2;
		labels2 = Eigen::Vector<double, Ce_shi_num>::Zero(Ce_shi_num);
		read_TEST_Label("t10k-labels.idx1-ubyte", labels2);

		int success = 0;
		double rate = 0;

		INPUT input = Eigen::Matrix<double, 1, LENGTH>::Zero(1, LENGTH);
		OUTPUT a1 = Eigen::Matrix<double, FIRST_NUM, 1>::Zero(FIRST_NUM, 1);
		OUTPUT a2 = Eigen::Matrix<double, SECOND_NUM, 1>::Zero(SECOND_NUM, 1);


		for (int i = 0; i < Ce_shi_num; i++)
		{
			input = data2.row(i);
			input = input.transpose();
			ForwardSpread(w1, w2, input, a1, a2);
			if (judge(a2) == labels2(i))
				success += 1;
		}
		cout << success << endl;
		rate = success * 1.0 / Ce_shi_num;
		cout << "成功率是" << rate;

// 我们也通过EasyBmp去进行了一个手写框的输入，并且进行了灰度值转化，但是由于Minist数据集中是对现实中
	//手写图片的数字识别，并且进行了数字的一定处理，比如将数字提取在图片正中央以及一些灰度值的识别，
	//而我们在用EasyBmp的一个手写框输入的时候，灰度值是固定的，而且由于时间不够，并没有进行过对于图片的一
	//些基本的处理，所以现实用鼠标书写的识别正确率欠佳。以下是我们之前的代码，
	/*
	INPUT pic = Eigen::Matrix<double, 1, LENGTH>::Zero(1, LENGTH);
	BIAS a1 = Eigen::Matrix<double, FIRST_NUM, 1>::Zero(FIRST_NUM, 1);
	BIAS a2 = Eigen::Matrix<double, SECOND_NUM, 1 >::Zero(SECOND_NUM, 1);


	/*
	pic = pic.transpose();
	BMP a;
	a.ReadFromFile("D:\\MFC大作业\\JIE_KOU_SHI_BIE\\JIE_KOU_SHI_BIE\\res\\bitmap1.bmp");
	double b[28][28] = { 0 };
	for (int i = 0; i < 28; i++)
		for (int j = 0; j < 28; j++)
		{
			int c[3] = { 0 };
			c[0] = a(i, j)->Red;
			c[1] = a(i, j)->Green;
			c[2] = a(i, j)->Blue;
			b[j][i] = (255 - (c[0] * 0.299 + c[1] * 0.587 + c[2] * 0.114))/180.0;

		}
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			pic(j + 28 * i, 0) = b[i][j];
			cout << setw(4) << b[i][j];
		}
		cout << endl;
	}
	ForwardSpread(w1, w2, pic, a1, a2);
	cout << "结果是" << judge(a2) << endl;*/
	return 0;
}
