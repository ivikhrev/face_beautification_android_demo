package com.example.test;

import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.graphics.Rect;
import android.os.Bundle;
import android.graphics.Bitmap;
import android.Manifest;
import android.os.SystemClock;
import android.util.Log;
import android.content.pm.PackageManager;
import android.view.WindowManager;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;
import java.io.BufferedReader;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.StandardCopyOption;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;


public class MainActivity extends CameraActivity implements CvCameraViewListener2 {

    private static final String TAG = "APP";
    private BlazeFace faceDetector;
    private FaceMesh landmarksDetector;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        try{
            System.loadLibrary(OPENCV_LIBRARY_NAME);
            Log.i(TAG, "Load OpenCV library");
        } catch (UnsatisfiedLinkError e) {
            Log.e("UnsatisfiedLinkError",
                    "Failed to load native OpenVINO libraries\n" + e.toString());
            System.exit(1);
        }

        String modelFile = "";
        try {
            modelFile = getResourcePath(getAssets().open("face_detection_short_range.tflite"), "face_detection_short_range", "tflite");
        } catch (IOException ex) {
        }
        faceDetector = new BlazeFace(modelFile, -1);

        try {
            modelFile = getResourcePath(getAssets().open("face_landmark.tflite"), "face_landmark", "tflite");
        } catch (IOException ex) {
        }
        landmarksDetector = new FaceMesh(modelFile, -1);

        if(checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 0);
        } else {
            setupCamera();
        }
    }

    private static String getResourcePath(InputStream in, String name, String ext) {
        String path = "";
        try {
            Path plugins = Files.createTempFile(name, ext);
            Files.copy(in, plugins, StandardCopyOption.REPLACE_EXISTING);
            path = plugins.toString();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return path;
    }

    private void setupCamera() {
        // Set up camera listener.
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraPermissionGranted();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults.length > 0 && grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            Log.e("PermissionError", "The application can't work without camera permissions");
            System.exit(1);
        }
        setupCamera();
    }

    @Override
    public void onResume() {
        super.onResume();
        mOpenCvCameraView.enableView();
    }
    @Override
    public void onCameraViewStarted(int width, int height) {}

    @Override
    public void onCameraViewStopped() {}

    private long prevFrameTime = 0;
    private long newFrameTime = 0;

    protected final float[] ptransform(Bitmap bitmap, FaceMeshMData mdata, int imageWidth, int imageHeight, int inputWidth, int inputHeight) {
        Rect faceRect = FaceMesh.enlargeFaceRoi(mdata.faceRect, imageWidth, imageHeight);
        int faceRoiWidth = faceRect.width();
        int faceRoiHeight = faceRect.height();
        PointF rotationCenter = new PointF((faceRect.left + faceRect.right) * 0.5f, (faceRect.top + faceRect.bottom) * 0.5f);
        double rotationRad = FaceMesh.calculateRotationRad(mdata.leftEye, mdata.rightEye);

        float[] dstPoints = {0, 0,
                inputWidth, 0,
                inputWidth, inputHeight,
                0, inputHeight};
        float[] srcPoints = {faceRect.left, faceRect.top,
                faceRect.right, faceRect.top,
                faceRect.right, faceRect.bottom,
                faceRect.left, faceRect.bottom};

        srcPoints = FaceMesh.rotatePoints(srcPoints, rotationRad, rotationCenter);

//        Matrix m = new Matrix();
//        m.setPolyToPoly(srcPoints, 0, dstPoints, 0, dstPoints.length >> 1);
//        Bitmap dstBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888);
//        Canvas canvas = new Canvas(dstBitmap);
//        canvas.clipRect(0, 0, inputWidth, inputHeight);
//        canvas.drawBitmap(bitmap, m, null);

        return srcPoints; // dstBitmap;
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        long newFrameTime = SystemClock.elapsedRealtime();
        Mat frame = inputFrame.rgba();
        Bitmap bmp = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(frame, bmp);
//        Imgproc.putText(frame, "HELLO WORLD", new Point(10, 40),
//                Imgproc.FONT_HERSHEY_COMPLEX, 1.8, new Scalar(0, 255, 0), 6);
        ArrayList<BBox> boxes = faceDetector.run(bmp, null);
        Bitmap res = Bitmap.createBitmap(196, 196, Bitmap.Config.ARGB_8888);
        Mat f = new Mat();
        for (BBox b : boxes) {
            Rect faceRect = FaceMesh.enlargeFaceRoi(b.face, frame.width(), frame.height());
            Imgproc.rectangle(frame, new Point(b.face.left, b.face.top), new Point(b.face.right, b.face.bottom), new Scalar(0,0,0), 2);
            Imgproc.rectangle(frame, new Point(faceRect.left, faceRect.top), new Point(faceRect.right,faceRect.bottom), new Scalar(255,0,0), 2);
            Imgproc.circle(frame, new Point(b.leftEye.x, b.leftEye.y), 2, new Scalar(0,255,255), -1);
            Imgproc.circle(frame, new Point(b.rightEye.x, b.rightEye.y), 2, new Scalar(0,255,255), -1);
            Imgproc.circle(frame, new Point(b.mouth.x, b.mouth.y), 2, new Scalar(0,255,255), -1);
            Imgproc.circle(frame, new Point(b.nose.x, b.nose.y), 2, new Scalar(0,255,255), -1);
            FaceMeshMData mdata = new FaceMeshMData();
            mdata.faceRect = b.face;
            mdata.leftEye = b.leftEye;
            mdata.rightEye = b.rightEye;
            FacialLandmarks lms = landmarksDetector.run(bmp, mdata);
            float[] r = ptransform(bmp, mdata, frame.width(), frame.height(), 192, 192);
//            double rotationRad = FaceMesh.calculateRotationRad(mdata.leftEye, mdata.rightEye);
//            RotatedRect rotRect = new RotatedRect(new Point(b.face.left, b.face.top), new Size(b.face.right - b.face.left, b.face.bottom - b.face.top), rotationRad);
            for (int j = 0; j < r.length / 2; j++) {
                Point first = new Point(r[2 * j], r[2 * j + 1]);
                Point second = new Point(r[2 * (j + 1) % 8], r[(2 * (j + 1) + 1) % 8]);
                Imgproc.line(frame, first, second, new Scalar(244,255,255), 2);
            }
//            Utils.bitmapToMat(res, f);
//            Mat roi = frame.submat(new org.opencv.core.Rect(0, 0, 196, 196));
//            f.copyTo(roi);
            for (PointF p : lms.faceOval) {
                Imgproc.circle(frame, new Point(p.x, p.y), 2, new Scalar(0,255,255), -1);
            }
        }

        double fps = 1000.f / (newFrameTime - prevFrameTime);
        prevFrameTime = newFrameTime;
        Imgproc.putText(frame, String.format("%.2f", fps)  + " FPS", new Point(10, 40), Imgproc.FONT_HERSHEY_COMPLEX, 1.8, new Scalar(100, 100, 120), 6);
        return frame;
    }
    public static final String OPENCV_LIBRARY_NAME = "opencv_java4";
    private CameraBridgeViewBase mOpenCvCameraView;
}