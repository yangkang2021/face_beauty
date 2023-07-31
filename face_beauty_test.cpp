#define  _CRT_SECURE_NO_WARNINGS

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>

#define HAVE_OPENCV

#include "face_beauty_neon.h"

using namespace cv;
using namespace std;


#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define CLIP3(x, a, b) MIN2(MAX2(a,x), b)

#define WRITE_FILE(FILE_NAME,DATA,LEN)                      \
{                                                           \
    static FILE* file = NULL;                               \
    const char *filename = FILE_NAME;                       \
    static int64_t total_size = 0;                          \
    if (!file) {                                            \
        remove(filename);                                   \
        file = fopen(filename, "wb");                       \
        if (!file)                                          \
            printf("Could not open %s\n", filename);             \
    }                                                       \
    if (file){                                              \
        fwrite(DATA, 1, LEN, file);                         \
        fflush(file);                                       \
        total_size += LEN;                                  \
        printf("write to %s(%lld):%d\n",filename,total_size,LEN);\
    }                                                       \
}


static uint8_t* yuv420p_to_rgb24(unsigned char* y_data, unsigned char* u_data, unsigned char* v_data, int ystride, int ustride, int vstride, int w, int h);

//6组测试图片
#if 0
//测试图片1
const char* yuv_file_name = "data/a-480x640.yuv";  //10ms
int w = 480;
int h = 640;
//[beauty] :(480 x 640) time ==> 10.7ms(93 fps) = 0.0(prepare)+0.6(integral)+5.8(lmsd)+1.9(sharpen)+1.7(boxblur)+0.8(blend)

#elif 0
//测试图片2
const char* yuv_file_name = "data/b-1200x1200.yuv"; //41-43ms
int w = 1200;
int h = 1200;


#elif 0
//测试图片3
const char* yuv_file_name = "data/c-640x360.yuv";//7ms
int w = 640;
int h = 360;

#elif 1
//测试图片1
const char* yuv_file_name = "data/a-1080x1920.yuv";  //10ms
int w = 1080;
int h = 1920;

#elif 0
//测试图片2
const char* yuv_file_name = "data/b-1920x1080.yuv"; //41-43ms
int w = 1920;
int h = 1080;

#elif 0
//测试图片3
const char* yuv_file_name = "data/c-1920x1080.yuv";//7ms
int w = 1920;
int h = 1080;
#endif

const char* output_filename = "data/beauty-dst.yuv";

int main(int argc, char** argv) 
{
	//读取yuv文件
	FILE* yuvfile = fopen(yuv_file_name, "rb");
	if (!yuvfile)
	{
		printf("open yuvfile [%s] error\n", yuv_file_name);
		return -1;
	}

	// 计算出文件的大小
	fseek(yuvfile, 0, SEEK_END);
	long file_size = ftell(yuvfile);
	fseek(yuvfile, 0, SEEK_SET);

	//将文件的内容读入内存
	unsigned char* y = (unsigned char*)malloc(file_size);
	unsigned char* u = y + w*h;
	unsigned char* v = u + w*h/4;

	int readed = 0;
	while (readed < file_size)
	{
		int nReadSize = fread(y + readed, 1, MIN2(1024, file_size - readed), yuvfile);
		if (nReadSize < 0) {

			printf("read yuvfile [%d/%d] error\n", nReadSize, file_size);
			fclose(yuvfile);
			free(y);
			return -1;
		}
		readed += nReadSize;
	}
	fclose(yuvfile);
	//free(yuv);
	printf("read yuvfile succeed:%s\n", yuv_file_name);
	assert(file_size == w * h * 3 / 2);
	
	FaceBeauty facebeauty;
	//运行一次，看效果
	bool result = facebeauty.beauty(y, u, v, w, w / 2, w / 2, w, h, 50, 50);
    //输出文件
	WRITE_FILE(output_filename, y, file_size);

	if (result) {//显示过程
		namedWindow("skin_mask", WINDOW_NORMAL);
		resizeWindow("skin_mask", cv::Size(600, 600));
		imshow("skin_mask", Mat(h, w, CV_8UC1, facebeauty.skin_mask));

		namedWindow("skin_mask_boxblur", WINDOW_NORMAL);
		resizeWindow("skin_mask_boxblur", cv::Size(600, 600));
		imshow("skin_mask_boxblur", Mat(h, w, CV_8UC1, facebeauty.skin_mask_boxblur));

		namedWindow("y_denoise", WINDOW_NORMAL);
		resizeWindow("y_denoise", cv::Size(600, 600));
		imshow("y_denoise", Mat(h, w, CV_8UC3, yuv420p_to_rgb24(facebeauty.y_denoise, u, v, w, w / 2, w / 2, w, h)));

		namedWindow("y_sharpen", WINDOW_NORMAL);
		resizeWindow("y_sharpen", cv::Size(600, 600));
		imshow("y_sharpen", Mat(h, w, CV_8UC3, yuv420p_to_rgb24(facebeauty.y_sharpen, u, v, w, w / 2, w / 2, w, h)));

		//namedWindow("src_bgr", WINDOW_NORMAL);
		//resizeWindow("src_bgr", cv::Size(600, 600));
		//imshow("src_bgr", Mat(h, w, CV_8UC3, facebeauty.src_bgr));

		system((string("start ffplay -window_title src -x 600 -y 600 -f rawvideo -video_size ") + to_string(w) + "x" + to_string(h) + " -pix_fmt yuv420p -i " + yuv_file_name).c_str());
	}

	//显示结果
	system((string("start ffplay -window_title dst -x 600 -y 600 -f rawvideo -video_size ") + to_string(w) + "x" + to_string(h) + " -pix_fmt yuv420p -i " + output_filename).c_str());

	//运行N次,计算一下均值
	const static int beauty_count = 50; //循环次数
	long long start_time = getTickCount();
	int remain_count = beauty_count;
	while (remain_count-- > 0)
	{
		facebeauty.beauty(y, u, v, w, w / 2, w / 2, w, h, 50, 50);
	}
	auto end_time = getTickCount();
	printf("LMSD with integral %d avg(%d x %d) time ==> %.1fms\n", beauty_count, w, h, (end_time - start_time) / getTickFrequency() * 1000 / beauty_count);

	waitKey(0);
	return 0;
}

static void YUV2RGB(int Y, int U, int V, unsigned char* R, unsigned char* G, unsigned char* B)
{
	*R = CLIP3(Y + 1.140 * V, 0, 255);
	*G = CLIP3(Y - 0.395 * U - 0.581 * V, 0, 255);
	*B = CLIP3(Y + 2.032 * U, 0, 255);

	//*R = CLIP3((100 * Y + 114 * V) / 100, 0, 255);
	//*G = CLIP3((1000 * Y - 395 * U - 581 * V) / 1000, 0, 255);
	//*B = CLIP3((1000 * Y + 2032 * U) / 1000, 0, 255);
};


static uint8_t* yuv420p_to_rgb24(unsigned char* y_data, unsigned char* u_data, unsigned char* v_data, int ystride, int ustride, int vstride,int w,int h)
{
	uint8_t* bgr = (uint8_t*)malloc(w * h * 3);
	int pos_y = 0;
	int pos_u = 0;
	int pos_v = 0;
	for (int cy = 0; cy < h; cy++)
	{
		auto yy = cy * ystride;
		auto uu = (cy / 2) * ustride;
		auto vv = (cy / 2) * vstride;
		for (int cx = 0; cx < w; cx++)
		{
			pos_y = yy + cx;
			pos_u = uu + (cx >> 1);
			pos_v = vv + (cx >> 1);

			auto Y = y_data[pos_y];
			auto U = u_data[pos_u];
			auto V = v_data[pos_v];

			unsigned char R, G, B;
			YUV2RGB(Y, U - 128, V - 128, &R, &G, &B);
			bgr[pos_y * 3 + 0] = B;
			bgr[pos_y * 3 + 1] = G;
			bgr[pos_y * 3 + 2] = R;
		}
	}
	return bgr;
}