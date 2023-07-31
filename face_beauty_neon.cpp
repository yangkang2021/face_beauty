#include "face_beauty_neon.h"
#include <memory>
#include <string.h>
#include <cassert>

#define LOG_TIME
//#define NO_NEON

#if defined(NO_NEON)
#else
    #if defined(_MSC_VER)
        //https://blog.csdn.net/XiaoHeiBlack/article/details/80988375
        #undef _MSC_VER //临时取消 _MSC_VER宏
        #if defined(_MSC_VER)|| defined (__INTEL_COMPILER)   //使该条件不满足
            #define _NEON2SSE_PERFORMANCE_WARNING(function, EXPLANATION) __declspec(deprecated(EXPLANATION)) function
        #if defined(_M_X64)
            #define _NEON2SSE_64BIT  _M_X64
        #endif
        #else
            #define _NEON2SSE_PERFORMANCE_WARNING(function, explanation)  function
        #endif

        #include "NEON_2_SSE.h"
    #else
        #include <arm_neon.h>
    #endif

#endif // defined(NO_NEON)

#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define CLIP3(x, a, b) MIN2(MAX2(a,x), b)

#if defined(NO_NEON)
#else
//只取src的前4个通道
inline float32x4_t uint8x8_to_float32x4(uint8x8_t src);
//输出只有前4个通道有效
inline uint8x8_t float32x4_to_uint8x8(float32x4_t f32src);
#endif

long long getTickCount();
float getTickFrequency();
void GetGrayIntegralImage(unsigned char* Src, float* Integral, float* IntegralSqr, int Width, int Height, int Stride, int Integral_Stride);
void GetGrayIntegralImage(unsigned char* Src, float* Integral, int Width, int Height, int Stride, int Integral_Stride);
void YUV2RGB(int Y, int U, int V, unsigned char* R, unsigned char* G, unsigned char* B);

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

bool FaceBeauty::beauty(unsigned char* y_data, unsigned char* u_data, unsigned char* v_data,
    int ystride, int ustride, int vstride,
    int w, int h, int skin_soft, int skin_white, bool simplified)
{
    return false;
    if (this->skin_soft_param <= 0)return false;

    //开始美颜
#if defined(LOG_TIME)
    auto start_time = getTickCount();
#endif
    //0.准备
    prepare(w, h, skin_soft, skin_white);

#if defined(LOG_TIME)
    auto prepare_time = getTickCount();
#endif

    //1. 皮肤检测
    skin_detect(y_data, u_data, v_data, ystride, ustride, vstride);

#if defined(LOG_TIME)
    auto skin_det_cost = getTickCount();
#endif

    //2.1 为皮肤mask的box滤波计算积分图
    GetGrayIntegralImage(skin_mask, y_integral_sum, w, h, ystride, y_integral_stride);

#if defined(LOG_TIME)
    auto boxblur_integral_time = getTickCount();
#endif
    //2.2 皮肤mask的box滤波
    skin_mask_blur();

#if defined(LOG_TIME)
    auto boxblur_time = getTickCount();
#endif

    //3.1. 为局部均方差滤波，计算和与平方和的积分图
    GetGrayIntegralImage(y_data, y_integral_sum, y_integral_sum_sqr, w, h, ystride, y_integral_stride);

#if defined(LOG_TIME)
    long long integral_time = getTickCount();
#endif
    //3.2. 局部均方差滤波
    lmsd(y_data, u_data, v_data, ystride, ustride, vstride);
#if defined(LOG_TIME)
    auto lmsd_cost = getTickCount();
#endif

    //4. 锐化均方差滤波结果
    sharpen();

#if defined(LOG_TIME)
    auto sharpen_time = getTickCount();
#endif

    //4. 均方差滤波与原图融合
    blend(y_data, ystride);

    //打印耗时
#if defined(LOG_TIME)
    auto bleand_time = getTickCount();
    FB_LOG("(%d x %d) time ==> %.1fms(%.0f fps) =  %.2f(prepare) + %.2f(skind) + %.2f(integral-s) + %.2f(boxblur) + %.2f(integral-s-sq) + %.2f(lmsd) + %.2f(sharpen) + %.2f(blend)",
        w, h,
        (bleand_time - start_time) / getTickFrequency() * 1000,
        1000.0f / ((bleand_time - start_time) / getTickFrequency() * 1000),
        (prepare_time - start_time) / getTickFrequency() * 1000,
        (skin_det_cost - prepare_time) / getTickFrequency() * 1000,
        (boxblur_integral_time - skin_det_cost) / getTickFrequency() * 1000,
        (boxblur_time - boxblur_integral_time) / getTickFrequency() * 1000,
        (integral_time - boxblur_time) / getTickFrequency() * 1000,
        (lmsd_cost - integral_time) / getTickFrequency() * 1000,
        (sharpen_time - lmsd_cost) / getTickFrequency() * 1000,
        (bleand_time - sharpen_time) / getTickFrequency() * 1000);
#endif
    return true;
}

void FaceBeauty::resetSkinWhitLUT()
{
    for (int i = 0; i < 256; i++)
    {
        float midtone = 0.667 * (1 - ((i - 127.0) / 127) * ((i - 127.0) / 127));
        skin_white_LUT[i] = CLIP3(i + skin_white_param / 2 * midtone, 0, 255);
    }
}

void FaceBeauty::resetCache()
{
    if (y_integral_sum_padding != nullptr) free(y_integral_sum_padding);
    if (y_integral_sum_sqr_padding != nullptr) free(y_integral_sum_sqr_padding);
    if (y_denoise != nullptr) free(y_denoise);
    if (y_sharpen != nullptr) free(y_sharpen);
    if (skin_mask != nullptr) free(skin_mask);
    if (skin_mask_boxblur != nullptr) free(skin_mask_boxblur);
    if (src_bgr != nullptr) free(src_bgr);
    if (w <= 0 || h <= 0)return;


    //两个积分图的缓存
    int integral_w = w + 1 + y_integral_padding * 2;
    int integral_h = h + 1 + y_integral_padding * 2;
    y_integral_sum_padding =     (float*)malloc(sizeof(float) * integral_w * integral_h); //包含了padding
    y_integral_sum_sqr_padding = (float*)malloc(sizeof(float) * integral_w * integral_h); //包含了padding
    
    memset(y_integral_sum_padding,     0x80, sizeof(float) * integral_w * integral_h);
    memset(y_integral_sum_sqr_padding, 0x80, sizeof(float) * integral_w * integral_h);

    y_integral_sum = y_integral_sum_padding + y_integral_stride * lmsd_radius + lmsd_radius;
    y_integral_sum_sqr = y_integral_sum_sqr_padding + y_integral_stride * lmsd_radius + lmsd_radius;

    y_denoise = (unsigned char*)malloc(sizeof(unsigned char) * w * h);
    y_sharpen = (unsigned char*)malloc(sizeof(unsigned char) * w * h);
    skin_mask = (unsigned char*)malloc(sizeof(unsigned char) * w * h);
    skin_mask_boxblur = (unsigned char*)malloc(sizeof(unsigned char) * w * h);
    src_bgr = (unsigned char*)malloc(sizeof(unsigned char) * w * h * 3);
}

inline void FaceBeauty::prepare(int w, int h, int skin_soft, int skin_white)
{
    //高宽必须是4的倍数
    if (w % 4 != 0 || h % 4 != 0)return;

    //面积太大无法处理,太小无需处理
    int area = w * h;
    if (area < 128 * 128 && area>1920 * 1080)return;

    //计算参数
    lmsd_radius = MAX2(w, h) * 0.02;

    y_integral_padding = lmsd_radius;
    y_integral_stride = w + y_integral_padding * 2 + 1;


    //高宽变化动态调整缓存
    if (this->w != w || this->h != h)
    {
        this->w = w;
        this->h = h;
        resetCache();
    }

    //调整参数范围
    if (this->skin_soft_param != skin_soft) this->skin_soft_param = CLIP3(skin_soft, 0, 100);
    if (this->skin_white_param != skin_white) {
        this->skin_white_param = CLIP3(skin_white, 0, 100);
        resetSkinWhitLUT();
    }
}

inline void FaceBeauty::skin_detect(unsigned char* y_data, unsigned char* u_data, unsigned char* v_data, int ystride, int ustride, int vstride)
{
#if defined(NO_NEON)
    memset(skin_mask, 0, w * h);

    for (int cy = 0; cy < h; cy++)
    {
        auto yy =       cy * ystride;
        auto uu = (cy / 2) * ustride;
        auto vv = (cy / 2) * vstride;
        for (int cx = 0; cx < w; cx++)
        {
            int pos_y = yy + cx;
            int pos_u = uu + (cx >> 1);
            int pos_v = vv + (cx >> 1);

            auto Y = y_data[pos_y];
            auto U = u_data[pos_u];
            auto V = v_data[pos_v];
#if 1
            if (Y > 80 && 77 < U && U < 135 && 133 < V && V < 180) {
                skin_mask[pos_y] = 255;
            }
#else       //RGB的皮肤检测，用来对比结果
            unsigned char R, G, B;
            YUV2RGB(Y, U - 128, V - 128, &R, &G, &B);
            src_bgr[pos_y * 3 + 0] = B;
            src_bgr[pos_y * 3 + 1] = G;
            src_bgr[pos_y * 3 + 2] = R;
            if (((R > 95) && (G > 40) && (B > 20) && (R > G) && (R > B) && (MAX2(R, G, B) - MIN2(R, G, B) > 15) && (abs(R - G) > 15)))
            {
                skin_mask[pos_y] = 255;
            }
#endif
        }
    }
#else //会出现黑线

    //133≤Cr≤173，77≤Cb≤127 
    //133≤V≤173，77≤U≤127 

    //比较用的常量
    uint8x16_t u80  = vdupq_n_u8(80);
    uint8x16_t u77  = vdupq_n_u8(77);
    uint8x16_t u135 = vdupq_n_u8(127);
    uint8x16_t u133 = vdupq_n_u8(133);
    uint8x16_t u180 = vdupq_n_u8(173);
    uint8x16_t ZERO = vdupq_n_u8(0); //16个V

    for (int cy = 0; cy < h-1; cy++) //最后一行不处理，保证neon每行的并行计算不会越界
    {
        auto y_line = y_data + cy * ystride;
        auto u_line = u_data + (cy >> 1) * ustride;
        auto v_line = v_data + (cy >> 1) * vstride;

        auto skin_mask_line = skin_mask + cy * ystride;

        for (int cx = 0; cx < w; cx += 64) //一次处理64个Y，16个U，16个V
        {
            uint8x16x4_t Y; 

            //读取16个U,V
            int cx_uv = cx >> 1;
            uint8x16_t U = vld1q_u8(u_line + cx_uv);
            uint8x16_t V = vld1q_u8(v_line + cx_uv);

            //计算UV是否在范围内
            auto CU = vandq_u8(vcgtq_u8(U, u77), vcltq_u8(U, u133));
            auto CV = vandq_u8(vcgtq_u8(V, u135), vcltq_u8(V, u180));

            //处理64个Y
            int parallel = 0;
            for (size_t k = 0; k < 4; k++) {
                //读取16个Y
                Y.val[k] = vld1q_u8(y_line + cx + parallel);
                
                //计算Y是否在范围内
                auto CY = vcgtq_u8(Y.val[k], u80);

                //写入结果
                uint8x16_t com_dst = vandq_u8(vandq_u8(CY, CU), CV);
                vst1q_u8(skin_mask_line + cx + parallel, com_dst);
                parallel += 16;
            }
        }
    }
#endif // defined(NO_NEONE)
}

inline void FaceBeauty::skin_mask_blur()
{
    int Radius = skin_mask_boxblur_radius;//至少10
    int Width = w;
    int Height = h;
    

#if 1//defined(NO_NEON)
    //#pragma omp parallel for
    for (int Y = 0; Y < Height; Y++)
    {
        int Y1 = Y - Radius;
        int Y2 = Y + Radius + 1;
        if (Y1 < 0) Y1 = 0;
        if (Y2 > Height) Y2 = Height;

        unsigned char* LinePD = skin_mask_boxblur + Y * Width;

        float* LineP1 = y_integral_sum + Y1 * y_integral_stride;
        float* LineP2 = y_integral_sum + Y2 * y_integral_stride;
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
#else //neon算法有问题,用cpu版本
    float32x4_t tl, tr, bl, br;
    float num_recip = 1.0f / ((2 * Radius+1) * (2 * Radius + 1));

    for (int Y = 0; Y < Height-1; Y++)
    {
        int y1 = Y - Radius;
        int y2 = Y + Radius +1;

        unsigned char* LinePD = skin_mask_boxblur + Y * Width;
        float* LineP1 = y_integral_sum + y1 * y_integral_stride;
        float* LineP2 = y_integral_sum + y2 * y_integral_stride;

        for (int X = 0; X < Width; X += 4)
        {
            int x1 = X - Radius;
            int x2 = X + Radius +1;

            tl = vld1q_f32(LineP1 + x1);
            bl = vld1q_f32(LineP2 + x1);
            tr = vld1q_f32(LineP1 + x2);
            br = vld1q_f32(LineP2 + x2);

            auto s = vsubq_f32(vaddq_f32(br, tl), vaddq_f32(bl, tr));
            auto mean = vmulq_n_f32(s, num_recip);

            float32_t newY_c[4] = {};
            vst1q_f32(newY_c, mean);
            auto Pdst = LinePD + X;
            for (int i = 0; i < 4; i++) {
                Pdst[i] = newY_c[i];
            }

            //if (1) {

            //    float32_t sum_c[4] = {};
            //    vst1q_f32(sum_c, s);
            //    for (size_t i = 0; i < 4; i++)
            //    {
            //        float Sum = LineP2[X2+i] - LineP1[X2 + i] - LineP2[X1 + i] + LineP1[X1 + i];
            //        int PixelCount = (X2 - X1) * (Y2 - Y1);					//	有效的像素数
            //        int dst = (Sum + (PixelCount >> 1)) / PixelCount;		//	四舍五入

            //        assert(Sum == sum_c[i]);
            //        assert(dst == Pdst[i]);
            //    }
            //}
        }
    }
#endif
}

inline void FaceBeauty::lmsd(unsigned char* y_data, unsigned char* u_data, unsigned char* v_data,int ystride, int ustride, int vstride)
{
    //memset(y_denoise, 0, w * h); 
#if defined(NO_NEON)
    int sigma2 = this->skin_soft_param * 5;

    int num, x1, y1, x2, y2;
    float s, var, mean, v, k;
    int tl, tr, bl, br;
    float tl2, tr2, bl2, br2;
    int const sum_w = w + 1;

    int pos_y;
    unsigned char Y;
    int newY;

    for (int cy = 0; cy < h; cy++) {

        for (int cx = 0; cx < w; cx++) {

			// 这个条件有问题
            if (cx>0 && cx<w &&
                skin_mask_boxblur[cy * w + cx - 1] == 0 &&
                skin_mask_boxblur[cy * w + cx + 0] == 0 &&
                skin_mask_boxblur[cy * w + cx + 1] == 0)
            {
                memcpy(y_denoise + pos_y, y_data + pos_y, 1);
                continue;
            }

            x1 = (cx - lmsd_radius) < 0 ? 0 : (cx - lmsd_radius);
            y1 = (cy - lmsd_radius) < 0 ? 0 : (cy - lmsd_radius);
            x2 = (cx + lmsd_radius) > w ? w : (cx + lmsd_radius);
            y2 = (cy + lmsd_radius) > h ? h : (cy + lmsd_radius);

            num = (x2 - x1) * (y2 - y1);
            tl = y_integral_sum[y1 * sum_w + x1];
            bl = y_integral_sum[y2 * sum_w + x1];
            tr = y_integral_sum[y1 * sum_w + x2];
            br = y_integral_sum[y2 * sum_w + x2];
            s = (br - bl - tr + tl);

            tl2 = y_integral_sum_sqr[y1 * sum_w + x1];
            bl2 = y_integral_sum_sqr[y2 * sum_w + x1];
            tr2 = y_integral_sum_sqr[y1 * sum_w + x2];
            br2 = y_integral_sum_sqr[y2 * sum_w + x2];
            var = (br2 - bl2 - tr2 + tl2);

            // 计算系数K
            mean = s / num;
            v = var / num - mean * mean;
            k = v / (v + sigma2);

            pos_y = cy * ystride + cx;
            Y = y_data[pos_y];

            newY = CLIP3((1 - k) * mean + k * Y, 0, 255);
            y_denoise[pos_y] = skin_white_LUT[newY];
        }
    }

#else
    float32x4_t sigma2 = vdupq_n_f32(this->skin_soft_param * 5);
    float num_rec = 1.0f / (2 * lmsd_radius * 2 * lmsd_radius);
    float32x4_t tl, tr, bl, br;
    float32x4_t s, var, mean, v, k;
    const int kernal_stride = lmsd_radius * y_integral_stride;
    uint8_t zeros[4] = { 0 };
    float newY_c[8] = {0};

    for (int cy = 0; cy < h-1; cy++)  //最后一行不处理
    {
        auto src_line      = y_data + cy * ystride;
        auto skin_line     = skin_mask_boxblur + cy * w;
        auto dst_line      = y_denoise + cy * w;
        auto pos           = cy * y_integral_stride;
        auto sum_row_p     = y_integral_sum     + pos;
        auto sum_sqr_row_p = y_integral_sum_sqr + pos;

        for (int cx = 0; cx < w; cx += 4) 
        {
            //if (cy == h - 1 && cx > w - 8)continue; //最后一行的最后4个像素不处理,因为vld1_u8读至少8个要溢出
            
            //TODO:只有这些地方需要做均方差滤波：皮肤区域，后面锐化需要的区域
            //if (memcmp(skin_line + cx, zeros, 4) == 0)continue;

            auto px1 = cx - kernal_stride - lmsd_radius;
            auto px2 = cx - kernal_stride + lmsd_radius;
            auto px3 = cx + kernal_stride - lmsd_radius;
            auto px4 = cx + kernal_stride + lmsd_radius;

            tl = vld1q_f32(sum_row_p + px1);
            bl = vld1q_f32(sum_row_p + px2);
            tr = vld1q_f32(sum_row_p + px3);
            br = vld1q_f32(sum_row_p + px4);
            s = vsubq_f32(vaddq_f32(br, tl), vaddq_f32(bl, tr));

            tl = vld1q_f32(sum_sqr_row_p + px1);
            bl = vld1q_f32(sum_sqr_row_p + px2);
            tr = vld1q_f32(sum_sqr_row_p + px3);
            br = vld1q_f32(sum_sqr_row_p + px4);
            var = vsubq_f32(vaddq_f32(br, tl), vaddq_f32(bl, tr));

            // 计算系数K
            //mean = s / num;
            //v = var / num - mean * mean;
            //k = v / (v + sigma2);
            //newY = (1 - k) * mean + k * Y; 

            mean = vmulq_n_f32(s, num_rec);
            v = vmlsq_f32(vmulq_n_f32(var, num_rec), mean, mean);
            k = vmulq_f32(v, vrecpeq_f32(vaddq_f32(v, sigma2)));

            auto Y = vld1_u8(src_line + cx);//vld1_u8图像末尾可能越界
            auto new_y_float =vmlaq_f32(vmlsq_f32(mean, k, mean), k, uint8x8_to_float32x4(Y));

            vst1q_f32(newY_c, new_y_float);
            for (int i = 0; i < 4 /*&& (cx+i <w)*/; i++) {
                dst_line[cx + i] = skin_white_LUT[CLIP3(int(newY_c[i]),0,255)];
            }
        }
    }
#endif
}

inline void FaceBeauty::sharpen()
{
#if defined(NO_NEON)
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
#else
    const static int STEP = 8;
    memcpy(y_sharpen, y_denoise, w);
    for (int j = 1; j < h - 1; j++) {
        int pos_line = j * w;
        int pos = 0;
        y_sharpen[pos_line] = y_denoise[pos_line];

        for (int i = 1; i < w - 1; i += STEP) {

            pos = pos_line + i;

            uint8x8_t src = vld1_u8(y_denoise + pos);

            int16x8_t dst = vmulq_n_s16(vreinterpretq_s16_u16(vmovl_u8(src)), 8);
            dst = vsubq_s16(dst, vreinterpretq_s16_u16(vmovl_u8(vld1_u8(y_denoise + pos - w))));
            dst = vsubq_s16(dst, vreinterpretq_s16_u16(vmovl_u8(vld1_u8(y_denoise + pos - 1))));
            dst = vsubq_s16(dst, vreinterpretq_s16_u16(vmovl_u8(vld1_u8(y_denoise + pos + 1))));
            dst = vsubq_s16(dst, vreinterpretq_s16_u16(vmovl_u8(vld1_u8(y_denoise + pos + w))));
            dst = vsubq_s16(dst, vreinterpretq_s16_u16(vmovl_u8(vld1_u8(y_denoise + pos - 1 - w))));
            dst = vsubq_s16(dst, vreinterpretq_s16_u16(vmovl_u8(vld1_u8(y_denoise + pos + 1 - w))));
            dst = vsubq_s16(dst, vreinterpretq_s16_u16(vmovl_u8(vld1_u8(y_denoise + pos - 1 + w))));
            dst = vsubq_s16(dst, vreinterpretq_s16_u16(vmovl_u8(vld1_u8(y_denoise + pos + 1 + w))));
            vst1_u8(y_sharpen + pos, vadd_u8(src, vqshrun_n_s16(dst, 2)));
        }
    }
#endif // defined(NO_NEON)
}

inline void FaceBeauty::blend(unsigned char* y_data, int ystride)
{
#if defined(NO_NEON)
#else
    uint8x8_t zero = vdup_n_u8(0);
    uint8x8_t c_255 = vdup_n_u8(255);
#endif

    for (int Y = 0; Y < h; Y++)
    {
        unsigned char* LinePS = y_sharpen + Y * w;
        unsigned char* LinePD = y_data + Y * ystride;
        unsigned char* LinePM = skin_mask_boxblur + Y * w;
        int X = 0;
        int Alpha, InvertAlpha;
#if defined(NO_NEON)
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
#else
        for (X = 0; X < w; X += 8)
        {
            auto S = vld1_u8(LinePS);
            auto D = vld1_u8(LinePD);
            auto M = vld1_u8(LinePM);

            uint16x8_t sum = vaddq_u16(vmull_u8(S, M), vmull_u8(D, vsub_u8(c_255, M)));
            auto dst = vqrshrn_n_u16(sum, 8);

            //auto dst = vbsl_u8(vqrshrn_n_u16(vaddq_u16(vmull_u8(S, M), vmull_u8(D, vsub_u8(c_255, M))), 8), D, M);
            vst1_u8(LinePD, dst);

            LinePS += 8;
            LinePD += 8;
            LinePM += 8;
        }
#endif
    }
}

void GetGrayIntegralImage(unsigned char* Src, float* Integral, float* IntegralSqr, int Width, int Height, int Stride, int Integral_Stride)
{
    //	你所看到的每一句代码都是作者辛勤劳作和多年经验的积累，希望你能尊重作者的成果
    //	你的每一个  评论  和  打赏  都是作者撰写更多博文和分享经验的鼓励。
    //	本代码对应博文见：http://www.cnblogs.com/Imageshop/p/6219990.html

    memset(Integral,    0, (Width + 1) * sizeof(float));					//	第一行都为0
    memset(IntegralSqr, 0, (Width + 1) * sizeof(float));					//	第一行都为0
    
    for (int Y = 0; Y < Height; Y++)
    {
        unsigned char* LinePS = Src + Y * Stride;

        auto L = Y * (Integral_Stride)+1;
        auto D = L + Integral_Stride; //(Y + 1) * (Integral_Stride)+1;

        float* LinePL = Integral + L;		//	上一行位置
        float* LinePD = Integral + D;		//	当前位置，注意每行的第一列的值都为0
        LinePD[-1] = 0;						//	第一列的值为0

        float* LinePLSqr = IntegralSqr + L;	//	上一行位置
        float* LinePDSqr = IntegralSqr + D;	//	当前位置，注意每行的第一列的值都为0
        LinePDSqr[-1] = 0;					//	第一列的值为0

        float Sum = 0;
        float SumSqr = 0;

        for (int X = 0; X < Width; X++)
        {
            Sum += LinePS[X];										//	行方向累加
            LinePD[X] = LinePL[X] + Sum;							//	更新积分图

            SumSqr += LinePS[X] * LinePS[X];						//	行方向累加
            LinePDSqr[X] = LinePLSqr[X] + SumSqr;				    //	更新积分图
        }
    }
}

void GetGrayIntegralImage(unsigned char* Src, float* Integral, int Width, int Height, int Stride, int Integral_Stride)
{
    //	你所看到的每一句代码都是作者辛勤劳作和多年经验的积累，希望你能尊重作者的成果
    //	你的每一个  评论  和  打赏  都是作者撰写更多博文和分享经验的鼓励。
    //	本代码对应博文见：http://www.cnblogs.com/Imageshop/p/6219990.html

    memset(Integral, 0, (Width + 1) * sizeof(float));					//	第一行都为0
    for (int Y = 0; Y < Height; Y++)
    {
        unsigned char* LinePS = Src + Y * Stride;
        float* LinePL = Integral + Y * (Integral_Stride ) + 1;				//	上一行位置
        float* LinePD = Integral + (Y + 1) * (Integral_Stride) + 1;		//	当前位置，注意每行的第一列的值都为0
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

#include <chrono>
#include <ctime>
using namespace std::chrono;

long long getTickCount() {
    system_clock::duration duration_since_epoch = system_clock::now().time_since_epoch(); // 从1970-01-01 00:00:00到当前时间点的时长
    time_t microseconds_since_epoch = duration_cast<microseconds>(duration_since_epoch).count(); // 将时长转换为微秒数
    return microseconds_since_epoch;
}

float getTickFrequency()
{
    return 1000000.0f;
}


#if defined(NO_NEON)
#else
//只取src的前4个通道
inline float32x4_t uint8x8_to_float32x4(uint8x8_t src)
{

    //1 u8x8->u16x8
    uint16x8_t u16x8src = vmovl_u8(src);

    //2 u16x8 -> u32x4high, u32x4low
    //uint32x4_t u32x4srch = vmovl_u16(vget_high_u16(u16x8src));
    uint32x4_t u32x4srcl = vmovl_u16(vget_low_u16(u16x8src));

    return vcvtq_f32_u32(u32x4srcl);
}

//输出只有前4个通道有效
inline uint8x8_t float32x4_to_uint8x8(float32x4_t f32src)
{
    uint32x4_t u32x4 = vcvtq_u32_f32(f32src);

    uint16x4_t u16x4 = vmovn_u32(u32x4);

    uint16x4_t c = vdup_n_u16(0);
    uint8x8_t u8x4 = vqmovn_u16(vcombine_u16(u16x4, c));
    return u8x4;
}
#endif
