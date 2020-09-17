#include "iostream"
#include "fstream"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

class MySVM : public CvSVM
{
public:
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}
	float get_rho()
	{
		return this->decision_func->rho;
	}
};

int main()
{
	int DescriptorDim;
 	MySVM svm;//SVM分类器
 	svm.load("SVM_HOG.xml");
 	DescriptorDim = svm.get_var_count();
	int supportVectorNum = svm.get_support_vector_count();
	cout<<"支持向量个数："<<supportVectorNum<<endl;
	//初始化
	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);
	//将支持向量的数据复制到supportVectorMat矩阵中
 	for(int i=0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针
		for(int j=0; j<DescriptorDim; j++)
		{
			supportVectorMat.at<float>(i,j) = pSVData[j];
		}
	}
  	//将alpha向量的数据复制到alphaMat中
	double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
	for(int i=0; i<supportVectorNum; i++)
		alphaMat.at<float>(0,i) = pAlphaData[i];
	//计算-(alphaMat * supportVectorMat),结果放到resultMat中
	resultMat = -1 * alphaMat * supportVectorMat;
	vector<float> myDetector;
	//将resultMat中的数据复制到数组myDetector中
	for(int i=0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0,i));
	}
	//最后添加偏移量rho，得到检测子
	myDetector.push_back(svm.get_rho());
	cout<<"检测子维数："<<myDetector.size()<<endl;
	//设置HOGDescriptor的检测子
	HOGDescriptor myHOG;
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	//保存检测子参数到文件
	ofstream fout("HOGDetectorForOpenCV.txt");
	for(int i=0; i<myDetector.size(); i++)
	{
		fout<<myDetector[i]<<endl;
	}
  /***************调用摄像头对目标进行检测识别*************/
	VideoCapture capture(0);
	while(1)//循环显示每一帧
	{
		Mat src;
		capture >> src;
		vector <Rect> found,found_filtered;//矩形框数组
		//cout<<"进行多尺度目标检测"<<endl;
		myHOG.detectMultiScale(src,found,0,Size(8,8),Size(32,32),1.05,2);//步长为8，图片每次放大比例为1.05，多次扫描
		//cout<<"找到矩形框个数："<<found.size()<<endl;
		for (int i=0;i<found.size();i++)
		{
			Rect r=found[i];
			int j=0;
			for (;j<found.size();j++)
				if (j!=i && (r & found[j]) == r)
				{
					break;
        }
			if (j == found.size())
			{
				found_filtered.push_back(r);
			}
		}
		/**画矩形框**/
		for (int i=0;i<found_filtered.size();i++)
		{
			Rect r=found_filtered[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(src, r.tl(), r.br(), Scalar(0,255,0), 3);
			//输出目标中心坐标到文件
			//double core_x=r.x+r.width;
			//double core_y=r.y+r.height;
			//ofstream corecout("D:\\tests\\object\\core.txt");
			//corecout<<"core_x"<<endl<<r.x<<"core_y"<<r.y<<endl;
			//corecout.close();
		}
		imshow("效果图",src);
		waitKey(30);
	}
		system("pause");
}
