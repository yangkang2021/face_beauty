#include "face_beauty_cpu.h"
#include <memory>
#include <string.h>

#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define CLIP3(x, a, b) MIN2(MAX2(a,x), b)

void GetGrayIntegralImage(unsigned char* Src, int* Integral, float* IntegralSqr, int Width, int Height, int Stride);
void GetGrayIntegralImage(unsigned char* Src, int* Integral, int Width, int Height, int Stride);

FaceBeauty::FaceBeauty()
{
    resetSkinWhitLUT();
}

FaceBeauty::~FaceBeauty()
{
    w = 0;
    h = 0;
    resetCache();
}

void FaceBeauty::beauty(
                        unsigned char * y_data, unsigned char * u_data, unsigned char * v_data,
                        int ystride, int ustride, int vstride,
                        int w, int h, int skin_soft, int skin_white, bool simplified)
{
    //开始美颜
#if defined(LOG_TIME)
    auto start_time = getTickCount();
#endif

    //高宽必须是4的倍数
    if(w%4 !=0 || h%4 !=0)return;

    //面积太大无法处理,太小无需处理
    int area = w*h;
    if(area <128*128 && area>1920*1080 )return;

    //高宽变化动态调整缓存
    if(this->w * this->h != area)
    {
        this->w = w;
        this->h = h;
        resetCache();
    }

    //调整参数范围
    if(this->skin_soft_param != skin_soft) this->skin_soft_param  = CLIP3(skin_soft,0,100);
    if(this->skin_white_param!= skin_white){
        this->skin_white_param = CLIP3(skin_white,0,100);
        resetSkinWhitLUT();
    }

    //准备好参数
    const int radius = MAX2(w, h) * 0.02;
    float sigma2 = this->skin_soft_param * 5;

#if defined(LOG_TIME)
    auto prepare_time = getTickCount();
#endif

    //1. 计算和与平方和的积分图
    GetGrayIntegralImage(y_data, y_integral_sum, y_integral_sum_sqr, w, h, ystride);

#if defined(LOG_TIME)
    long long integral_time = getTickCount();
#endif

    //2. 局部均方差滤波 + 美白 + 皮肤检测
    memset(skin_mask, 0, w * h); //只有它需要置零

    int num, x1, y1, x2, y2;
    float s, var, mean, v, k;
    int tl, tr, bl, br;
    float tl2, tr2, bl2, br2;
    int const sum_w = w + 1;

    int pos;
    int pos_u;
    int pos_v;
    unsigned char Y,U,V;
    int newY;

    for (int cy = 0; cy < h; cy++) {

        for (int cx = 0; cx < w; cx++) {
            x1 = (cx - radius) < 0 ? 0 : (cx - radius);
            y1 = (cy - radius) < 0 ? 0 : (cy - radius);
            x2 = (cx + radius) > w ? w : (cx + radius);
            y2 = (cy + radius) > h ? h : (cy + radius);

            num = (x2 - x1) * (y2 - y1);
            //s     = get_block_sum(sum, x1, y1, x2, y2, 0);
            tl = y_integral_sum[y1 * sum_w + x1];
            bl = y_integral_sum[y2 * sum_w + x1];
            tr = y_integral_sum[y1 * sum_w + x2];
            br = y_integral_sum[y2 * sum_w + x2];
            s = (br - bl - tr + tl);

            //var   = get_block_sqr_sum(sqrsum, x1, y1, x2, y2, 0);
            tl2 = y_integral_sum_sqr[y1 * sum_w + x1];
            bl2 = y_integral_sum_sqr[y2 * sum_w + x1];
            tr2 = y_integral_sum_sqr[y1 * sum_w + x2];
            br2 = y_integral_sum_sqr[y2 * sum_w + x2];
            var = (br2 - bl2 - tr2 + tl2);

            // 计算系数K
            mean = s / num;
            v = var / num - mean * mean;
            k = v / (v + sigma2);

            pos         = cy * ystride + cx;
            pos_u       = (cy / 2) * ustride + cx / 2;
            pos_v       = (cy / 2) * vstride + cx / 2;
            Y           = y_data[pos];

            newY = CLIP3((1 - k) * mean + k * Y, 0 ,255);
            y_denoise[pos] = skin_white_LUT[newY];

            if(simplified)continue;

            //皮肤检测
            U = u_data[pos_u];
            V = v_data[pos_v];
            if ((Y > 80) && (77 < U && U < 135) && (133 < V && V < 180)){
                skin_mask[pos] = 255;
            }
            /*unsigned char R, G, B;
            YUV2RGB(Y, U-128, V - 128, &R, &G, &B);
            bgr.data[pos * 3 + 0] = B;
            bgr.data[pos * 3 + 1] = G;
            bgr.data[pos * 3 + 2] = R;
            if (((R > 95) && (G > 40) && (B > 20) && (R > G) && (R > B) && (MAX2(R, G, B) - MIN2(R, G, B) > 15) && (abs(R - G) > 15)))
            {
                skin_mask[pos] = 255;
            }*/
        }
    }

#if defined(LOG_TIME)
    auto lmsd_cost = getTickCount();
#endif

    //3. 锐化均方差滤波结果
    if(simplified){
        for (int i = 0; i<h; i++) {
            memcpy(y_data + i * ystride, y_denoise + i*w, w);
        }
#if defined(LOG_TIME)
        auto sharpen_time = getTickCount();
        //打印耗时
        FB_LOG("LMSD with integral(%d x %d) time(ms) ==> %.1f(total--%.0f fps) =  %.1f(prepare) + %.1f(integral) + %.1f(lmsd) + %.1f(copy)\n",
            w, h,
            (sharpen_time - start_time) / getTickFrequency() * 1000,
            1000.0f / ((sharpen_time - start_time) / getTickFrequency() * 1000),
            (prepare_time - start_time) / getTickFrequency() * 1000,
            (integral_time - prepare_time) / getTickFrequency() * 1000,
            (lmsd_cost - integral_time) / getTickFrequency() * 1000,
            (sharpen_time - lmsd_cost) / getTickFrequency() * 1000);
#endif
        return;
    }
    for (int j = 1; j < h - 1; j++) {
        for (int i = 1; i < w - 1; i++) {
            int pos = i + j * w;
            y_sharpen[pos] = CLIP3(y_denoise[pos] +
                (y_denoise[pos] * 8
                    - y_denoise[pos - w]
                    - y_denoise[pos - 1]
                    - y_denoise[pos + 1]
                    - y_denoise[pos + w]
                    - y_denoise[pos - 1 - w]
                    - y_denoise[pos + 1 - w]
                    - y_denoise[pos - 1 + w]
                    - y_denoise[pos + 1 + w]
                    ) / 4, 0, 255);
        }
    }

#if defined(LOG_TIME)
    auto sharpen_time = getTickCount();
#endif

    //4. 皮肤mask的box滤波
    {
        int Radius = 10;//至少10
        int Width = w;
        int Height = h;
        GetGrayIntegralImage(skin_mask, y_integral_sum, Width, Height, Width);
        //#pragma omp parallel for
        for (int Y = 0; Y < Height; Y++)
        {
            int Y1 = Y - Radius;
            int Y2 = Y + Radius + 1;
            if (Y1 < 0) Y1 = 0;
            if (Y2 > Height) Y2 = Height;

            int* LineP1 = y_integral_sum + Y1 * (Width + 1);
            int* LineP2 = y_integral_sum + Y2 * (Width + 1);

            unsigned char* LinePD = skin_mask_boxblur + Y * Width;
            for (int X = 0; X < Width; X++)
            {
                int X1 = X - Radius;
                if (X1 < 0) X1 = 0;
                int X2 = X + Radius + 1;
                if (X2 > Width) X2 = Width;

                int Sum = LineP2[X2] - LineP1[X2] - LineP2[X1] + LineP1[X1];
                int PixelCount = (X2 - X1) * (Y2 - Y1);					//	有效的像素数
                LinePD[X] = (Sum + (PixelCount >> 1)) / PixelCount;		//	四舍五入
            }
        }
    }
#if defined(LOG_TIME)
    auto boxblur_time = getTickCount(); 
#endif

    //5.均方差滤波与原图融合
    for (int Y = 0; Y < h; Y++)
    {
        unsigned char* LinePS = y_sharpen + Y * w;
        unsigned char* LinePD = y_data + Y * ystride;
        unsigned char* LinePM = skin_mask_boxblur + Y * w;
        int X = 0;
        int Alpha, InvertAlpha;
        for (X = 0; X < w; X++)
        {
            Alpha = 255 - LinePM[0];
            if (Alpha != 255)
            {
                #define Div255(V)  (((V >> 8) + V + 1) >> 8);
                InvertAlpha = 255 - Alpha;
                LinePD[0] = Div255(LinePD[0] * Alpha + LinePS[0] * InvertAlpha);
            }
            LinePS++;
            LinePD++;
            LinePM++;
        }
    }

#if defined(LOG_TIME)
    auto bleand_time = getTickCount();
    //打印耗时
    FB_LOG("LMSD with integral(%d x %d) time(ms) ==> %.1f(total--%.0f fps) =  %.1f(prepare) + %.1f(integral) + %.1f(lmsd) + %.1f(sharpen)+ %.1f(boxblur) + %.1f(blend)\n",
        w, h,
        (bleand_time - start_time) / getTickFrequency() * 1000,
        1000.0f / ((bleand_time - start_time) / getTickFrequency() * 1000),
        (prepare_time - start_time) / getTickFrequency() * 1000,
        (integral_time - prepare_time) / getTickFrequency() * 1000,
        (lmsd_cost - integral_time) / getTickFrequency() * 1000,
        (sharpen_time - lmsd_cost) / getTickFrequency() * 1000,
        (boxblur_time - sharpen_time) / getTickFrequency() * 1000,
        (bleand_time - boxblur_time) / getTickFrequency() * 1000);
#endif
}

void FaceBeauty::resetSkinWhitLUT()
{
    for (int i = 0; i < 256; i++)
    {
        float midtone = 0.667 * (1 - ((i - 127.0) / 127) * ((i - 127.0) / 127));
        skin_white_LUT[i] = CLIP3(i + skin_white_param/2 * midtone,0,255);
    }
}

void FaceBeauty::resetCache()
{
    if(y_integral_sum!=nullptr) free(y_integral_sum);
    if(y_integral_sum_sqr!=nullptr) free(y_integral_sum_sqr);
    if(y_denoise!=nullptr) free(y_denoise);
    if(y_sharpen!=nullptr) free(y_sharpen);
    if(skin_mask!=nullptr) free(skin_mask);
    if(skin_mask_boxblur!=nullptr) free(skin_mask_boxblur);
    if(w<=0 || h<=0)return;

    y_integral_sum = (int*)malloc(sizeof(int) * (w + 1) * (h + 1));
    y_integral_sum_sqr = (float*)malloc(sizeof(float) * (w + 1) * (h + 1));
    y_denoise = (unsigned char*)malloc(sizeof(unsigned char) * w * h);
    y_sharpen = (unsigned char*)malloc(sizeof(unsigned char) * w * h);
    skin_mask = (unsigned char*)malloc(sizeof(unsigned char) * w * h);
    skin_mask_boxblur = (unsigned char*)malloc(sizeof(unsigned char) * w * h);
}

void GetGrayIntegralImage(unsigned char* Src, int* Integral, float* IntegralSqr, int Width, int Height, int Stride)
{
    //	你所看到的每一句代码都是作者辛勤劳作和多年经验的积累，希望你能尊重作者的成果
    //	你的每一个  评论  和  打赏  都是作者撰写更多博文和分享经验的鼓励。
    //	本代码对应博文见：http://www.cnblogs.com/Imageshop/p/6219990.html

    memset(Integral,    0, (Width + 1) * sizeof(int));					//	第一行都为0
    memset(IntegralSqr, 0, (Width + 1) * sizeof(float));					//	第一行都为0
    for (int Y = 0; Y < Height; Y++)
    {
        unsigned char* LinePS = Src + Y * Stride;

        int* LinePL = Integral + Y * (Width + 1) + 1;				//	上一行位置
        int* LinePD = Integral + (Y + 1) * (Width + 1) + 1;			//	当前位置，注意每行的第一列的值都为0
        LinePD[-1] = 0;												//	第一列的值为0

        float* LinePLSqr = IntegralSqr + Y * (Width + 1) + 1;				//	上一行位置
        float* LinePDSqr = IntegralSqr + (Y + 1) * (Width + 1) + 1;			//	当前位置，注意每行的第一列的值都为0
        LinePDSqr[-1] = 0;												//	第一列的值为0

        float  SumSqr = 0;
        for (int X = 0, Sum = 0; X < Width; X++)
        {
            Sum += LinePS[X];										//	行方向累加
            LinePD[X] = LinePL[X] + Sum;							//	更新积分图

            SumSqr += LinePS[X] * LinePS[X];						//	行方向累加
            LinePDSqr[X] = LinePLSqr[X] + SumSqr;						//	更新积分图
        }
    }
}

void GetGrayIntegralImage(unsigned char* Src, int* Integral, int Width, int Height, int Stride)
{
    //	你所看到的每一句代码都是作者辛勤劳作和多年经验的积累，希望你能尊重作者的成果
    //	你的每一个  评论  和  打赏  都是作者撰写更多博文和分享经验的鼓励。
    //	本代码对应博文见：http://www.cnblogs.com/Imageshop/p/6219990.html

    memset(Integral, 0, (Width + 1) * sizeof(int));					//	第一行都为0
    for (int Y = 0; Y < Height; Y++)
    {
        unsigned char* LinePS = Src + Y * Stride;
        int* LinePL = Integral + Y * (Width + 1) + 1;				//	上一行位置
        int* LinePD = Integral + (Y + 1) * (Width + 1) + 1;			//	当前位置，注意每行的第一列的值都为0
        LinePD[-1] = 0;												//	第一列的值为0
        for (int X = 0, Sum = 0; X < Width; X++)
        {
            Sum += LinePS[X];										//	行方向累加
            LinePD[X] = LinePL[X] + Sum;							//	更新积分图
        }
    }
}

void YUV2RGB(int Y, int U, int V, unsigned char* R, unsigned char* G, unsigned char* B)
{
    *R = CLIP3(Y + 1.140 * V, 0, 255);
    *G = CLIP3(Y - 0.395 * U - 0.581 * V, 0, 255);
    *B = CLIP3(Y + 2.032 * U, 0, 255);

    //*R = CLIP3((100 * Y + 114 * V) / 100, 0, 255);
    //*G = CLIP3((1000 * Y - 395 * U - 581 * V) / 1000, 0, 255);
    //*B = CLIP3((1000 * Y + 2032 * U) / 1000, 0, 255);
};

#if defined(HAVE_OPENCV) //#define HAVE_OPENCV
#else
    #include <chrono>
    #include <ctime>
    using namespace std::chrono;

    long long getTickCount(){
        system_clock::duration duration_since_epoch = system_clock::now().time_since_epoch(); // 从1970-01-01 00:00:00到当前时间点的时长
        time_t microseconds_since_epoch = duration_cast<microseconds>(duration_since_epoch).count(); // 将时长转换为微秒数
        return microseconds_since_epoch;
    }

    float getTickFrequency()
    {
        return 1000000.0f;
    }
#endif
