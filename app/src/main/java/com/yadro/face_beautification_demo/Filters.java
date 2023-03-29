package com.yadro.face_beautification_demo;

import static org.opencv.imgproc.Imgproc.Laplacian;
import static org.opencv.imgproc.Imgproc.fillPoly;
import static org.opencv.imgproc.Imgproc.medianBlur;

import android.graphics.PointF;
import android.graphics.Rect;

import org.checkerframework.checker.units.qual.A;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.List;

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

    private static MatOfPoint convertToMatOfPoints(ArrayList<PointF> arr) {
        ArrayList<Point> ocvPoints = new ArrayList<>();
        for (PointF p : arr) {
            ocvPoints.add(new Point(p.x, p.y));
            Point p1 = new Point();
        }
        MatOfPoint mop = new MatOfPoint();
        mop.getNativeObjAddr();
        mop.fromList(ocvPoints);
//        ArrayList<MatOfPoint> res = new ArrayList<MatOfPoint>();
//        res.add(mop);
        return mop;
    }

    private static native void beautifyFace_(long img_nativeObj, long faceOval_nativeObj, long faceFeatures_nativeObj);

    public static void beautifyFace(Mat img, ArrayList<FacialLandmarks> landmarks) {
        ArrayList<MatOfPoint> faceOval = new ArrayList<>();
        ArrayList<MatOfPoint> faceFeatures = new ArrayList<>();
        for (FacialLandmarks faceLms : landmarks) {
            faceOval.add(convertToMatOfPoints(faceLms.faceOval));

            faceFeatures.add(convertToMatOfPoints(faceLms.leftEye));
            faceFeatures.add(convertToMatOfPoints(faceLms.leftBrow));
            faceFeatures.add(convertToMatOfPoints(faceLms.rightEye));
            faceFeatures.add(convertToMatOfPoints(faceLms.rightBrow));
            faceFeatures.add(convertToMatOfPoints(faceLms.nose));
            faceFeatures.add(convertToMatOfPoints(faceLms.lips));

        }

        List<Mat> faceOval_tmplm = new ArrayList<Mat>((faceOval != null) ? faceOval.size() : 0);
        Mat faceOval_mat = Converters.vector_vector_Point_to_Mat(faceOval, faceOval_tmplm);

        List<Mat> faceFeatures_tmplm = new ArrayList<Mat>((faceFeatures != null) ? faceFeatures.size() : 0);
        Mat faceFeatures_mat = Converters.vector_vector_Point_to_Mat(faceFeatures, faceFeatures_tmplm);

        beautifyFace_(img.getNativeObjAddr(), faceOval_mat.getNativeObjAddr(), faceFeatures_mat.getNativeObjAddr());
    }
}
