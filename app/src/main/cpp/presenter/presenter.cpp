#include <jni.h>

#include <opencv2/core/mat.hpp>

#include <monitors/presenter.h>

extern "C" JNIEXPORT jlong JNICALL
Java_com_yadro_face_1beautification_1demo_Presenter_getPresenter(JNIEnv * env, jobject jobj, jstring keysJObj) {
    const char* keys = env->GetStringUTFChars(keysJObj, 0);
    Presenter* presenter = new Presenter(keys);
    return (jlong)presenter;
}

extern "C" JNIEXPORT void JNICALL
Java_com_yadro_face_1beautification_1demo_Presenter_drawGraphs_1(JNIEnv * env, jobject jobj, jlong presenterAddr, jlong matAddr) {
    cv::Mat& frame = *((cv::Mat*)matAddr);
    Presenter* presenter = (Presenter*)presenterAddr;
    presenter->drawGraphs(frame);
}

extern "C" JNIEXPORT void JNICALL
Java_com_yadro_face_1beautification_1demo_Presenter_handleKey_1(JNIEnv * env, jobject jobj, jlong presenterAddr, jint key) {
    Presenter* presenter = (Presenter*)presenterAddr;
    presenter->handleKey((int)key);
}

JNIEXPORT void JNICALL
Java_com_yadro_face_1beautification_1demo_Presenter_delete(JNIEnv *, jobject, jlong presenterAddr) {
    Presenter* presenter = (Presenter*)presenterAddr;
    delete presenter;
}