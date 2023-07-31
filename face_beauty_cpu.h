#ifndef FACEBEAUTY_H
#define FACEBEAUTY_H

//#define LOG_TIME

class FaceBeauty
{
public:
    FaceBeauty();
    ~FaceBeauty();
    void beauty(
                unsigned char * y_data, unsigned char * u_data, unsigned char * v_data,
                int ystride, int ustride, int vstride,
                int w, int h,
                int skin_soft_param = 50, int skin_white_param = 50,bool simplified = false);

private:
    void resetSkinWhitLUT();
    void resetCache();
private:
    unsigned char skin_white_LUT[256] = { 0 };
    int skin_soft_param = 50;
    int skin_white_param = 50;

    int w = 0;
    int h = 0;

    //内存缓存
    int* y_integral_sum = nullptr;
    float* y_integral_sum_sqr = nullptr;
    unsigned char* y_denoise = nullptr;
    unsigned char* y_sharpen = nullptr;
public:
    unsigned char* skin_mask = nullptr;
    unsigned char* skin_mask_boxblur = nullptr;//只有它需要置零
};

#if defined(HAVE_OPENCV) //#define HAVE_OPENCV
    #include <opencv2/opencv.hpp>
    using namespace cv;
#else
    long long getTickCount();
    float getTickFrequency();
#endif

//日志函数LOG定义
//http://www.cnblogs.com/morewindows/archive/2011/08/18/2144112.html
//http://blog.csdn.net/cjh965063777/article/details/38868195
#if defined(ANDROID)
//android
#define TAG "FaceBeauty-JNI"
#include <android/log.h>
#define FB_LOG(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, TAG, "[]==>" fmt,##__VA_ARGS__)
#elif  defined(IOS)
//ios
#import <Foundation/Foundation.h>
#define FB_LOG(fmt, ...) NSLog((@"[%s:%d:%s]==>" @fmt @"\n"),  __FILE__, __LINE__, __PRETTY_FUNCTION__, ##__VA_ARGS__)
#elif  defined(MAC)
//mac
#define FB_LOG(...) {printf(__VA_ARGS__);printf("\n");}
#else
//other
#include <stdio.h>
#define FB_LOG(fmt, ...) {printf("[%s:%d:%s]==>" fmt "\n", __FILE__, __LINE__,__func__,__VA_ARGS__);}
#endif //WEBRTC_ANDROID

#if defined(NDEBUG)
#define FB_LOG_DEBUG(...)
#else
#define FB_LOG_DEBUG(...) FB_LOG(__VA_ARGS__)
#endif

#endif // FACEBEAUTY_H
