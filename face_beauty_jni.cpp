#include "com_cv_FaceBeauty.h"
#include <android/log.h>
#include "face_beauty.h"
#include <libyuv.h>
#include <memory>
#include "face_beauty_c.h"

extern "C" void JNIEXPORT JNICALL Java_com_cv_FaceBeauty_nativeFaceBeauty
(JNIEnv * env, jclass,
 jobject yByteBuf, jobject uByteBuf, jobject vByteBuf,
 jint ystride, jint ustride, jint vstride,
 jint width, jint height,
 jboolean beauty,int skin_soft, int skin_white, jboolean simplified,
 jboolean mirror)
{
    unsigned char * y = static_cast<uint8_t*>(env->GetDirectBufferAddress(yByteBuf));
    unsigned char * u = static_cast<uint8_t*>(env->GetDirectBufferAddress(uByteBuf));
    unsigned char * v = static_cast<uint8_t*>(env->GetDirectBufferAddress(vByteBuf));

    /*__android_log_print(ANDROID_LOG_ERROR,"FaceBeauty",
                        "%p--%p--%p, "
                        "%d--%d--%d, "
                        "%d--%d, "
                        "%d--%d--%d--%d,"
                        "--%d",
                        y,u,v,
                        ystride,ustride,vstride,
                        width,height,
                        beauty,skin_soft,skin_white,simplified,
                        mirror);*/

    face_beauty(y,u,v,ystride,ustride,vstride,width,height,beauty,skin_soft,skin_white,simplified,mirror);
}

extern "C" jint JNIEXPORT JNICALL JNI_OnLoad(JavaVM* jvm, void* reserved)
{
    return JNI_VERSION_1_6;
}

extern "C" void JNIEXPORT JNICALL JNI_OnUnload(JavaVM* jvm, void* reserved)
{
}
