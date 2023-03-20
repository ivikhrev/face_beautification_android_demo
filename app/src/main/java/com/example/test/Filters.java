package com.example.test;

import static org.opencv.core.Core.bitwise_and;
import static org.opencv.core.Core.bitwise_not;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2RGB;
import static org.opencv.imgproc.Imgproc.GaussianBlur;
import static org.opencv.imgproc.Imgproc.Laplacian;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.bilateralFilter;
import static org.opencv.imgproc.Imgproc.fillPoly;
import static org.opencv.imgproc.Imgproc.medianBlur;
import static org.opencv.imgproc.Imgproc.threshold;

import android.graphics.PointF;
import android.graphics.Rect;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

public class Filters {
    private static Mat fillPolyContours(int width, int height, ArrayList<MatOfPoint> contours) {
        Mat out = new Mat(height, width, CvType.CV_8UC1, new Scalar(0));
        fillPoly(out, contours, new Scalar(255, 255, 255));
        return out;
    }

    private static Mat unsharpMask(Mat src, final int blurKSize, final float strength) {
        Mat blurred = new Mat();
        medianBlur(src, blurred, blurKSize);
        Mat laplacian = new Mat();
        Laplacian(blurred, laplacian,  CvType.CV_8U);
        Core.multiply(laplacian, new Scalar(strength), laplacian);
        Core.subtract(src, laplacian, laplacian);
        return laplacian;
    }

    private static ArrayList<MatOfPoint> convertToMatOfPoints(ArrayList<PointF> arr) {
        ArrayList<Point> ocvPoints = new ArrayList<>();
        for (PointF p : arr) {
            ocvPoints.add(new Point(p.x, p.y));
        }
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(ocvPoints);
        ArrayList<MatOfPoint> res = new ArrayList<MatOfPoint>();
        res.add(mop);
        return res;
    }

    public static Mat beautifyFace(Mat img, Rect faceRect, FacialLandmarks landmarks) {
        ArrayList<MatOfPoint> faceOval = convertToMatOfPoints(landmarks.faceOval);
        ArrayList<MatOfPoint> faceFeatures = convertToMatOfPoints(landmarks.leftEye);
        faceFeatures.addAll(convertToMatOfPoints(landmarks.leftBrow));
        faceFeatures.addAll(convertToMatOfPoints(landmarks.rightEye));
        faceFeatures.addAll(convertToMatOfPoints(landmarks.rightBrow));
        faceFeatures.addAll(convertToMatOfPoints(landmarks.nose));
        faceFeatures.addAll(convertToMatOfPoints(landmarks.lips));

        Rect rect = FaceMesh.enlargeFaceRoi(faceRect, img.width(), img.height());
        org.opencv.core.Rect ocvRect = new org.opencv.core.Rect(rect.left, rect.top, rect.width(), rect.height());

        Mat mskSharp = fillPolyContours(img.width(), img.height(), faceFeatures);
        Mat mskSharpG = new Mat();
        GaussianBlur(mskSharp, mskSharpG, new Size(5, 5), 0.0);

        Mat mskBlur = fillPolyContours(img.width(), img.height(), faceOval);
        Mat mskBlurG = new Mat();
        GaussianBlur(mskBlur, mskBlurG, new Size(5, 5), 0.0);

        Mat mask = new Mat();
        mskBlurG.copyTo(mask, mskSharpG);
        Mat mskBlurFinal = new Mat();
        Core.subtract(mskBlurG, mask, mskBlurFinal);
        Mat mskFacesGaussed = new Mat();
        Core.add(mskBlurFinal, mskSharpG, mskFacesGaussed);

        Mat mskFacesWhite = new Mat();
        threshold(mskFacesGaussed, mskFacesWhite, 0, 255, THRESH_BINARY);
        Mat mskNoFaces = new Mat();
        bitwise_not(mskFacesWhite, mskNoFaces);

        final int pixelsDiameter = 15;
        final double sigmaColor = 37.5;
        final double sigmaSpace = 37.5;

        Mat imgBilat = img.clone();
//        auto time = std::chrono::steady_clock::now();
        bilateralFilter(img.submat(ocvRect), imgBilat.submat(ocvRect), pixelsDiameter, sigmaColor, sigmaSpace);
//          Imgproc.cvtColor(img.submat(ocvRect), imgBilat.submat(ocvRect), COLOR_BGR2RGB);
//        filterMetrics.update(time);

        final int blurKSize = 3;
        final float strengthCoeff = 0.7f;
        Mat imgSharp = unsharpMask(img, blurKSize, strengthCoeff);

        Mat imgBilatMasked = new Mat();
        bitwise_and(imgBilat, imgBilat, imgBilatMasked, mskBlurFinal);
        Mat imgSharpMasked = new Mat();
        bitwise_and(imgSharp, imgSharp, imgSharpMasked, mskSharpG);
        Mat imgInMasked = new Mat();
        bitwise_and(img, img, imgInMasked, mskNoFaces);

        Mat imgBeautif = new Mat();
        Core.add(imgBilatMasked, imgSharpMasked, imgBeautif);
        Core.add(imgBeautif, imgInMasked, imgBeautif);
        return imgBeautif;
    }
}
