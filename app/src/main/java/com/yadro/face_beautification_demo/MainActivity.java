package com.yadro.face_beautification_demo;

import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;

import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.Manifest;
import android.content.ContentValues;
import android.content.pm.PackageManager;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.OrientationEventListener;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import org.checkerframework.checker.units.qual.A;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity {
    private static final String[] CAMERA_PERMISSION = new String[]{Manifest.permission.CAMERA};
    private static final int CAMERA_REQUEST_CODE = 10;

    private PreviewView previewView;
    private SurfaceView surfaceView;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private CameraSelector cameraSelector;
    private int lensFacing = CameraSelector.LENS_FACING_FRONT;
    private final String LENS_FACING_KEY = "LENS_FACING";
    private TextView textView;

    private Presenter presenter;
    private final float IMAGE_ASPECT_RATIO = 16 / 9.f;
    private int surfaceRectRight = 0;
    private int surfaceRectBottom = 0;

    enum DisplayMode {
        ORIGINAL,
        LANDMARKS,
        FILTERED;
        private static final DisplayMode[] vals = values();

        public DisplayMode next() {
            return vals[(this.ordinal() + 1) % vals.length];

        }
    }
    private DisplayMode displayMode = DisplayMode.FILTERED;
    private static final String TAG = "APP";
    public static final String OPENCV_LIBRARY_NAME = "opencv_java4";
    public static final String FILTERS_LIBRARY_NAME = "filters";
    public static final String PRESENTER_LIBRARY_NAME = "presenter";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Recover the instance state.
        if (savedInstanceState != null) {
            lensFacing = savedInstanceState.getInt(LENS_FACING_KEY);
        }

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        try{
            System.loadLibrary(OPENCV_LIBRARY_NAME);
            Log.i(TAG, "Load OpenCV library");
        } catch (UnsatisfiedLinkError e) {
            Log.e("UnsatisfiedLinkError",
                    "Failed to load native OpenCV libraries\n" + e.toString());
            System.exit(1);
        }

        try{
            System.loadLibrary(FILTERS_LIBRARY_NAME);
            Log.i(TAG, "Load filters library");
        } catch (UnsatisfiedLinkError e) {
            Log.e("UnsatisfiedLinkError",
                    "Failed to load native filters libraries\n" + e.toString());
            System.exit(1);
        }

        try{
            System.loadLibrary(PRESENTER_LIBRARY_NAME);
            Log.i(TAG, "Load presenter library");
        } catch (UnsatisfiedLinkError e) {
            Log.e("UnsatisfiedLinkError",
                    "Failed to load native filters libraries\n" + e.toString());
            System.exit(1);
        }

        presenter = new Presenter("cdm");
        setContentView(R.layout.activity_main);

        surfaceView = findViewById(R.id.surfaceView);
        surfaceView.setZOrderMediaOverlay(true);

        if (!hasCameraPermission()) {
            requestCameraPermission();
        }
        startCamera();

        ImageButton switchCamera = findViewById(R.id.switchCamera);
        switchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (hasCameraPermission()) {
                    switchCamera();
                } else {
                    requestCameraPermission();
                }
            }
        });

        Button changeDisplayMode = findViewById(R.id.changeDisplayMode);
        changeDisplayMode.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                displayMode = displayMode.next();
            }
        });
    }

//    @Override
//    public void onRestoreInstanceState(Bundle savedInstanceState) {
//        lensFacing = savedInstanceState.get;
//    }

    @Override
    public void onSaveInstanceState(Bundle outState) {
        outState.putInt(LENS_FACING_KEY, lensFacing);;

        // Call superclass to save any view hierarchy.
        super.onSaveInstanceState(outState);
    }
    private boolean hasCameraPermission() {
        return ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestCameraPermission() {
        ActivityCompat.requestPermissions(
                this,
                CAMERA_PERMISSION,
                CAMERA_REQUEST_CODE
        );
    }

    private void switchCamera() {
        if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
            lensFacing = CameraSelector.LENS_FACING_BACK;
        }
        else if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            lensFacing = CameraSelector.LENS_FACING_FRONT;
        }
        startCamera();
    }

    private void startCamera() {
//        previewView = findViewById(R.id.previewView);
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
//        textView = findViewById(R.id.orientation);
        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    bindImageAnalysis(cameraProvider);
                } catch (ExecutionException | InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private static Bitmap toBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer buffer = planes[0].getBuffer();
        int pixelStride = planes[0].getPixelStride();
        int rowStride = planes[0].getRowStride();
        int rowPadding = rowStride - pixelStride * image.getWidth();
        Bitmap bitmap = Bitmap.createBitmap(image.getWidth()+rowPadding/pixelStride,
                image.getHeight(), Bitmap.Config.ARGB_8888);
        bitmap.copyPixelsFromBuffer(buffer);
        return Bitmap.createScaledBitmap(bitmap, 720,480,false);
    }

    private BlazeFace faceDetector;
    private FaceMesh landmarksDetector;
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
    private void initModels() {
        String modelFile = "";
        try {
            modelFile = getResourcePath(getAssets().open("face_detection_short_range.tflite"), "face_detection_short_range", "tflite");
        } catch (IOException ex) {
        }
        faceDetector = new BlazeFace(modelFile, "CPU", 4);

        try {
            modelFile = getResourcePath(getAssets().open("face_landmark.tflite"), "face_landmark", "tflite");
        } catch (IOException ex) {
        }
        landmarksDetector = new FaceMesh(modelFile, "GPU", 4);
    }
    private void drawLandmarks(Mat frame, ArrayList<PointF> landmarks) {
        ArrayList<Point> pts = new ArrayList<>();
        for (PointF p : landmarks) {
            pts.add(new Point(p.x, p.y));
            Imgproc.circle(frame, new Point(p.x, p.y), 2, new Scalar(225, 193, 110), -1);
        }
        MatOfPoint m = new MatOfPoint();
        m.fromList(pts);
        ArrayList<MatOfPoint> l = new ArrayList<MatOfPoint>();
        l.add(m);
        Imgproc.polylines(frame, l, true, new Scalar(192, 192, 192), 1);
    }
    private void renderResults(Mat frame, ArrayList<FacialLandmarks> allLanmdarks) {
        for (FacialLandmarks landmarks : allLanmdarks) {
            drawLandmarks(frame, landmarks.faceOval);
            drawLandmarks(frame, landmarks.leftEye);
            drawLandmarks(frame, landmarks.leftBrow);
            drawLandmarks(frame, landmarks.rightEye);
            drawLandmarks(frame, landmarks.rightBrow);
            drawLandmarks(frame, landmarks.nose);
            drawLandmarks(frame, landmarks.lips);
        }
    }
    private void drawImage(Bitmap bitmap) {
        Log.d(TAG, "DRAWING IMAGE size " + bitmap.getWidth() + "x" + bitmap.getHeight());
        Log.d(TAG, "DRAWING surface size " + surfaceView.getWidth() + "x" + surfaceView.getHeight());
        Rect src = new Rect(0, 0, bitmap.getWidth(), bitmap.getHeight());
        Rect dst;
        int w = surfaceView.getWidth();
        int h = surfaceView.getHeight();
        if (w < h) {
            h = Math.round(w * IMAGE_ASPECT_RATIO);
            dst = new Rect(0, 0, w, h);
        } else {
            h = Math.round(w / IMAGE_ASPECT_RATIO);
            dst = new Rect(0, 0, w, h);
        }
        SurfaceHolder holder = surfaceView.getHolder();
        Canvas canvas = holder.lockCanvas();
        canvas.drawBitmap(bitmap, src, dst, null);
        holder.unlockCanvasAndPost(canvas);

    }
    private void bindImageAnalysis(@NonNull ProcessCameraProvider cameraProvider) {
        if (faceDetector == null || landmarksDetector == null) {
            initModels();
        }

        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), new ImageAnalysis.Analyzer() {
            @Override
            @ExperimentalGetImage
            public void analyze(@NonNull ImageProxy imageProxy) {
                long newFrameTime = SystemClock.elapsedRealtime();
                Log.d(TAG, "starting, got  image " + imageProxy.getWidth() + "x" + String.valueOf(imageProxy.getHeight()));
                int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();
                Bitmap bmp = toBitmap(imageProxy.getImage());

                Log.d(TAG, "bitmap, got  image " + bmp.getWidth() + "x" + bmp.getHeight());
                Matrix matrix = new Matrix();
                matrix.postRotate((float)rotationDegrees);
                if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                    matrix.postScale(-1.f, 1.f);
                }
                bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true);
                Mat frame = new Mat();
                Utils.bitmapToMat(bmp, frame);
                Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
                Utils.matToBitmap(frame, bmp);
                ArrayList<BBox> boxes = faceDetector.run(bmp, null);
                ArrayList<FacialLandmarks> lms = new ArrayList<>();
                Log.d(TAG, "Detected " + boxes.size());
                for (BBox b : boxes) {
                    Rect faceRect = FaceMesh.enlargeFaceRoi(b.face, frame.width(), frame.height());
//                    Imgproc.rectangle(frame, new Point(b.face.left, b.face.top), new Point(b.face.right, b.face.bottom), new Scalar(0,0,0), 2);
//                    Imgproc.rectangle(frame, new Point(faceRect.left, faceRect.top), new Point(faceRect.right,faceRect.bottom), new Scalar(255,0,0), 2);
//                    Imgproc.circle(frame, new Point(b.leftEye.x, b.leftEye.y), 2, new Scalar(0,255,255), -1);
//                    Imgproc.circle(frame, new Point(b.rightEye.x, b.rightEye.y), 2, new Scalar(0,255,255), -1);
//                    Imgproc.circle(frame, new Point(b.mouth.x, b.mouth.y), 2, new Scalar(0,255,255), -1);
//                    Imgproc.circle(frame, new Point(b.nose.x, b.nose.y), 2, new Scalar(0,255,255), -1);
                    FaceMeshMData mdata = new FaceMeshMData();
                    mdata.faceRect = b.face;
                    mdata.leftEye = b.leftEye;
                    mdata.rightEye = b.rightEye;
                    lms.add(landmarksDetector.run(bmp, mdata));
//
                }
                if (!lms.isEmpty() && displayMode != DisplayMode.ORIGINAL) {
                    if (displayMode == DisplayMode.FILTERED)
                        Filters.beautifyFace(frame, lms);
                    if (displayMode == DisplayMode.LANDMARKS)
                        renderResults(frame, lms);
                }
                double fps = 1000.f / (SystemClock.elapsedRealtime() - newFrameTime);
                Imgproc.putText(frame, String.format("%.2f", fps)  + " FPS", new Point(10, 40), Imgproc.FONT_HERSHEY_COMPLEX, 1.8, new Scalar(100, 100, 120), 6);
//                Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
//                Utils.matToBitmap(frame, bmp);
                presenter.drawGraphs(frame);
                Utils.matToBitmap(frame, bmp);
                drawImage(bmp);
                imageProxy.close();
            }
        });

        Preview preview = new Preview.Builder().build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(lensFacing).build();

        try {
            cameraProvider.unbindAll();
            cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, imageAnalysis);
        } catch (Exception ex) {
            Log.e(TAG, "Failed to bind use case");
        }
    }
}