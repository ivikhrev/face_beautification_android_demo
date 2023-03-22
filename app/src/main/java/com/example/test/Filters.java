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

    private static ArrayList<MatOfPoint> convertToMatOfPoints(ArrayList<PointF> arr) {
        ArrayList<Point> ocvPoints = new ArrayList<>();
        for (PointF p : arr) {
            ocvPoints.add(new Point(p.x, p.y));
            Point p1 = new Point();
        }
        MatOfPoint mop = new MatOfPoint();
        mop.getNativeObjAddr();
        mop.fromList(ocvPoints);
        ArrayList<MatOfPoint> res = new ArrayList<MatOfPoint>();
        res.add(mop);
        return res;
    }

    private static native void beautifyFace_(long img_nativeObj, long faceOval_natieObj, long faceFeatures_nativeObj,
                                          int faceRectX, int faceRectY, int faceRectW, int faceRectH);
    public static void beautifyFace(Mat img, Rect faceRect, FacialLandmarks landmarks) {
        ArrayList<MatOfPoint> faceOval = convertToMatOfPoints(landmarks.faceOval);
        List<Mat> faceOval_tmplm = new ArrayList<Mat>((faceOval != null) ? faceOval.size() : 0);
        Mat faceOval_mat = Converters.vector_vector_Point_to_Mat(faceOval, faceOval_tmplm);

        ArrayList<MatOfPoint> faceFeatures = convertToMatOfPoints(landmarks.leftEye);
        faceFeatures.addAll(convertToMatOfPoints(landmarks.leftBrow));
        faceFeatures.addAll(convertToMatOfPoints(landmarks.rightEye));
        faceFeatures.addAll(convertToMatOfPoints(landmarks.rightBrow));
        faceFeatures.addAll(convertToMatOfPoints(landmarks.nose));
        faceFeatures.addAll(convertToMatOfPoints(landmarks.lips));
        List<Mat> faceFeatures_tmplm = new ArrayList<Mat>((faceFeatures != null) ? faceFeatures.size() : 0);
        Mat faceFeatures_mat = Converters.vector_vector_Point_to_Mat(faceFeatures, faceFeatures_tmplm);

        beautifyFace_(img.getNativeObjAddr(), faceOval_mat.getNativeObjAddr(), faceFeatures_mat.getNativeObjAddr(), faceRect.left,
                faceRect.top, faceRect.width(), faceRect.height());
    }
}
