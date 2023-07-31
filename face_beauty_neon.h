#ifndef FACEBEAUTY_H
#define FACEBEAUTY_H

class FaceBeauty
{
public:
    FaceBeauty();
    ~FaceBeauty();
    bool beauty(unsigned char * y_data, unsigned char * u_data, unsigned char * v_data,
                int ystride, int ustride, int vstride,
                int w, int h,
                int skin_soft_param = 50, int skin_white_param = 50,bool simplified = false);

private:
    void resetSkinWhitLUT();
    void resetCache();

private:
    void prepare(int w, int h, int skin_soft, int skin_white);
    void skin_detect(unsigned char* y_data, unsigned char* u_data, unsigned char* v_data, int ystride, int ustride, int vstride);
    void skin_mask_blur();
    void lmsd(unsigned char* y_data, unsigned char* u_data, unsigned char* v_data, int ystride, int ustride, int vstride);
    void sharpen();
    void blend(unsigned char* y_data, int ystride);

public:
    unsigned char* skin_mask = nullptr;         //需要置零
    unsigned char* skin_mask_boxblur = nullptr; //不需要置零，每个像素都会被复写
    unsigned char* y_denoise = nullptr;         //需要置零，有些位置的不会做lmsd
    unsigned char* y_sharpen = nullptr;
    unsigned char* src_bgr = nullptr;           //原图的bgr图，用于测试
public:

    //内存缓存
    float* y_integral_sum_padding = nullptr;
    float* y_integral_sum_sqr_padding = nullptr;

    float* y_integral_sum = nullptr;
    float* y_integral_sum_sqr = nullptr;

    int y_integral_padding = 0;
    int y_integral_stride = 0;

    unsigned char skin_white_LUT[256] = { 0 };
    int skin_soft_param = 50;
    int skin_white_param = 50;

    int w = 0;
    int h = 0;

    int lmsd_radius = 12;              //局部均方差滤波半径
    int sharpen_radius = 1;            //锐化滤波半径
    int skin_mask_boxblur_radius = 10; //皮肤检测mask的滤波半径
};


//日志函数LOG定义
//http://www.cnblogs.com/morewindows/archive/2011/08/18/2144112.html
//http://blog.csdn.net/cjh965063777/article/details/38868195
#if defined(ANDROID)
//android
#define TAG "FaceBeauty-JNI"
#include <android/log.h>
#define FB_LOG(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, TAG, "[]:" fmt,##__VA_ARGS__)
#elif  defined(IOS)
//ios
#import <Foundation/Foundation.h>
#define FB_LOG(fmt, ...) NSLog((@"[%s]:" @fmt @"\n"), __PRETTY_FUNCTION__, ##__VA_ARGS__)
#elif  defined(MAC)
//mac
#define FB_LOG(...) {printf(__VA_ARGS__);printf("\n");}
#else
//other
#include <stdio.h>
#define FB_LOG(fmt, ...) {printf("[%s]:" fmt "\n",__func__,__VA_ARGS__);}
#endif //WEBRTC_ANDROID

#if defined(NDEBUG)
#define FB_LOG_DEBUG(...)
#else
#define FB_LOG_DEBUG(...) FB_LOG(__VA_ARGS__)
#endif

#endif // FACEBEAUTY_H
