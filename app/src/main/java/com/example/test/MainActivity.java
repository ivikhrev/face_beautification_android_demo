package com.example.test;

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
import org.opencv.core.Mat;
import org.opencv.core.Point;
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

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;


public class MainActivity extends CameraActivity implements CvCameraViewListener2 {

    private static final String TAG = "APP";
    private BlazeFace faceDetector;
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

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(String modelPath) throws IOException {
        Log.i(TAG, "Reading model asset file: " + modelPath);
        File file=new File(modelPath);
        FileInputStream inputStream = new FileInputStream(file);
        FileChannel fileChannel = inputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
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


    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        long newFrameTime = SystemClock.elapsedRealtime();
        Mat frame = inputFrame.rgba();
        Bitmap bmp = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(frame, bmp);
//        Imgproc.putText(frame, "HELLO WORLD", new Point(10, 40),
//                Imgproc.FONT_HERSHEY_COMPLEX, 1.8, new Scalar(0, 255, 0), 6);
        ArrayList<BBox> boxes = faceDetector.run(bmp, null);
        for (BBox b : boxes) {
            Imgproc.rectangle(frame, new Point(b.face.left, b.face.top), new Point(b.face.right, b.face.bottom), new Scalar(244,255,255), 2);
            Imgproc.circle(frame, new Point(b.leftEye.x, b.leftEye.y), 2, new Scalar(0,255,255), -1);
            Imgproc.circle(frame, new Point(b.rightEye.x, b.rightEye.y), 2, new Scalar(0,255,255), -1);
            Imgproc.circle(frame, new Point(b.mouth.x, b.mouth.y), 2, new Scalar(0,255,255), -1);
            Imgproc.circle(frame, new Point(b.nose.x, b.nose.y), 2, new Scalar(0,255,255), -1);
        }
        double fps = 1000.f / (newFrameTime - prevFrameTime);
        prevFrameTime = newFrameTime;
        Imgproc.putText(frame, String.format("%.2f", fps)  + " FPS", new Point(10, 40), Imgproc.FONT_HERSHEY_COMPLEX, 1.8, new Scalar(100, 100, 120), 6);
        return frame;
    }
    public static final String OPENCV_LIBRARY_NAME = "opencv_java4";
    private CameraBridgeViewBase mOpenCvCameraView;
}