#include <jni.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {
    cv::UMat fillPolyContours(int width, int height, const std::vector<std::vector<cv::Point>>& contours) {
    cv::UMat out{height, width, CV_8UC1, cv::Scalar(0)};
    cv::fillPoly(out, contours, {255, 255, 255});
    return out;
}

inline cv::UMat unsharpMask(const cv::Mat& src, const int blurKSize, const float strength) {
    cv::UMat blurred;
    cv::medianBlur(src, blurred, blurKSize);
    cv::UMat laplacian;
    cv::Laplacian(blurred, laplacian, CV_8U);
    cv::multiply(laplacian, cv::Scalar(strength),laplacian);
    cv::subtract(src, laplacian, laplacian);
    return laplacian;
}

//vector_Point
void Mat_to_vector_Point(cv::Mat& mat, std::vector<cv::Point>& v_point)
{
    v_point.clear();
    if (!(mat.type()==CV_32SC2 && mat.cols==1)) {
        throw std::runtime_error("Wrong mat type");
    }
    v_point = (std::vector<cv::Point>) mat;
}

void Mat_to_vector_Mat(cv::Mat& mat, std::vector<cv::Mat>& v_mat) {
    v_mat.clear();
    if(mat.type() == CV_32SC2 && mat.cols == 1)
    {
        v_mat.reserve(mat.rows);
        for(int i=0; i<mat.rows; i++)
        {
            cv::Vec<int, 2> a = mat.at< cv::Vec<int, 2> >(i, 0);
            long long addr = (((long long)a[0])<<32) | (a[1]&0xffffffff);
            cv::Mat& m = *( (cv::Mat*) addr );
            v_mat.push_back(m);
        }
    } else {
        throw std::runtime_error("Mat_to_vector_Mat() FAILED: mat.type() == CV_32SC2 && mat.cols == 1");
    }
}

void Mat_to_vector_vector_Point(cv::Mat& mat, std::vector< std::vector<cv::Point > >& vv_pt) {
    std::vector<cv::Mat> vm;
    vm.reserve( mat.rows );
    Mat_to_vector_Mat(mat, vm);
    for(size_t i=0; i<vm.size(); i++) {
        std::vector<cv::Point> vpt;
        Mat_to_vector_Point(vm[i], vpt);
        vv_pt.push_back(vpt);
    }
}

}  // namespace

extern "C" JNIEXPORT void JNICALL
Java_com_yadro_face_1beautification_1demo_Filters_beautifyFace_1(JNIEnv *env, jclass jobj, jlong matAddr, jlong faceOvalAddr, jlong faceFeaturesAddr) {
    //    const cv::Mat& img, Rect faceRect, FacialLandmarks landmarks) {
    cv::Mat& img = *((cv::Mat*)matAddr);

    std::vector<std::vector<cv::Point>> faceOval;
    cv::Mat& faceOvalPtsMat = *((cv::Mat*)faceOvalAddr);
    Mat_to_vector_vector_Point( faceOvalPtsMat, faceOval );
//    Mat& img = *((Mat*)img_nativeObj);


    std::vector<std::vector<cv::Point>> faceFeatures;
    cv::Mat& faceFeaturesPtsMat = *((cv::Mat*)faceFeaturesAddr);
    Mat_to_vector_vector_Point( faceFeaturesPtsMat, faceFeatures );

//    cv::Rect rect(faceRectX, faceRectY, faceRectW, faceRectH);

//    std::vector<cv::Point> faceOval = face.landmarks.faceOval;
//    std::vector<std::vector<cv::Point>> faceFeatures = face.getFeatures();
//    auto rect = FaceMesh::enlargeFaceRoi(face.box) & cv::Rect({}, img.size());

    cv::UMat mskSharp = fillPolyContours(img.size().width, img.size().height, faceFeatures);
    cv::UMat mskSharpG;
    cv::GaussianBlur(mskSharp, mskSharpG, {5, 5}, 0.0);

    cv::UMat mskBlur = fillPolyContours(img.size().width, img.size().height, {faceOval});
    cv::UMat mskBlurG;
    cv::GaussianBlur(mskBlur, mskBlurG, {5, 5}, 0.0);
//
    cv::UMat mask;
    mskBlurG.copyTo(mask, mskSharpG);
    cv::UMat mskBlurFinal;// = mskBlurG - mask;
    cv::subtract(mskBlurG, mask, mskBlurFinal);
    cv::UMat mskFacesGaussed; //= mskBlurFinal + mskSharpG;
    cv::add(mskBlurFinal, mskSharpG, mskFacesGaussed);

    cv::UMat mskFacesWhite;
    cv::threshold(mskFacesGaussed, mskFacesWhite, 0, 255, cv::THRESH_BINARY);
    cv::UMat mskNoFaces;
    cv::bitwise_not(mskFacesWhite, mskNoFaces);

    cv::UMat imgBilat;
    img.copyTo(imgBilat);

    const int pixelsDiameter = 9;
    const double sigmaColor = 30.0, sigmaSpace = 30.0;

//    auto time = std::chrono::steady_clock::now();
    cv::bilateralFilter(img, imgBilat, pixelsDiameter, sigmaColor, sigmaSpace);
//    filterMetrics.update(time);
//
    const int blurKSize = 3;
    const float strengthCoeff = 0.7;
    cv::UMat imgSharp = unsharpMask(img, blurKSize, strengthCoeff);

    cv::UMat imgBilatMasked;
    cv::bitwise_and(imgBilat, imgBilat, imgBilatMasked, mskBlurFinal);
    cv::UMat imgSharpMasked;
    cv::bitwise_and(imgSharp, imgSharp, imgSharpMasked, mskSharpG);
    cv::UMat imgInMasked;
    cv::bitwise_and(img, img, imgInMasked, mskNoFaces);
//
    cv::UMat imgBeautif;// = imgBilatMasked + imgSharpMasked + imgInMasked;
    cv::add(imgSharpMasked, imgInMasked, imgBeautif);
    cv::UMat imgBeautif2;
    cv::add(imgBilatMasked, imgBeautif, imgBeautif2);
    img = imgBeautif2.getMat(cv::ACCESS_READ).clone();
}
