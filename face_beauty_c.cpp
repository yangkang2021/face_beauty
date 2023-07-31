#include "face_beauty_c.h"
#include <libyuv.h>
#include <memory>
#include "face_beauty.h"

FaceBeauty faceBeauty;
unsigned char *yuv_mirror_cache[3] = {nullptr};
int yuv_mirror_cache_len[3] = {0};

void face_beauty(
    const unsigned char * y, const unsigned char * u, const unsigned char * v,
    int ystride, int ustride, int vstride,
    int width, int height,
    bool beauty,int skin_soft, int skin_white, bool simplified,
    bool mirror)
{
    if(width<=0 || height<=0 || width%4 !=0 || height%4 !=0)return;
    if(beauty) faceBeauty.beauty(
                                 (unsigned char*)y,
                                 (unsigned char*)u,
                                 (unsigned char*)v,
                                 ystride,ustride,vstride,width,height,
                                 skin_soft,skin_white,simplified);

    if(!mirror)return;
    
#if defined(LOG_TIME)
    auto mirror_start = getTickCount();
#endif
    
    int yuv_plane_size[3] = {ystride * height,
                             ustride * height/2,
                             vstride * height/2};

    for(int i = 0;i<3;i++) {
        if (yuv_mirror_cache_len[i] < yuv_plane_size[i]) {
            if (yuv_mirror_cache[i])free(yuv_mirror_cache[i]);
            yuv_mirror_cache_len[i] = yuv_plane_size[i];
            yuv_mirror_cache[i] = (unsigned char *) malloc(yuv_mirror_cache_len[i]);
        }
        memset(yuv_mirror_cache[i], 0, yuv_mirror_cache_len[i]);
    }
   libyuv::I420Mirror(
            y, ystride,
            u, ustride,
            v, vstride,

            yuv_mirror_cache[0], ystride,
            yuv_mirror_cache[1], ustride,
            yuv_mirror_cache[2], vstride,
            width, height);

    //拷贝数据，两种方案
#if 0
    memcpy((void *)y, yuv_mirror_cache[0], yuv_plane_size[0]);
    memcpy((void *)u, yuv_mirror_cache[1], yuv_plane_size[1]);
    memcpy((void *)v, yuv_mirror_cache[2], yuv_plane_size[2]);
#else
    for (int i = 0; i< height; i++) {
        memcpy((unsigned char *)y + i * ystride, yuv_mirror_cache[0] + i*ystride, width);
    }
    
    for (int i = 0; i<height/2; i++) {
        memcpy((unsigned char *)u + i * ustride, yuv_mirror_cache[1] + i*ustride, width/2);
    }
    
    for (int i = 0; i<height/2; i++) {
        memcpy((unsigned char *)v + i * vstride, yuv_mirror_cache[2] + i*vstride, width/2);
    }
#endif
    
#if defined(LOG_TIME)
    auto mirror_end = getTickCount();
    LOG("mirror time:%f ms", (mirror_end - mirror_start)*1000/getTickFrequency());
#endif
}
