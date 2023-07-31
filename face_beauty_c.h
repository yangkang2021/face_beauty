#ifndef FACEBEAUTY_C_H
#define FACEBEAUTY_C_H

#ifdef __cplusplus
extern "C" {
#endif
 
void face_beauty(
    const unsigned char * y, const unsigned char * u, const unsigned char * v,
    int ystride, int ustride, int vstride,
    int width, int height,
    bool beauty,int skin_soft, int skin_white, bool simplified,
    bool mirror
);
 
#ifdef __cplusplus
}
#endif

#endif // FACEBEAUTY_C_H
