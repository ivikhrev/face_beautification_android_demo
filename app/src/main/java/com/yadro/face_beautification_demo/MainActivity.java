package com.yadro.face_beautification_demo;

import android.graphics.Bitmap;
import android.graphics.Canvas;

import android.graphics.Matrix;
import android.graphics.PointF;
import android.graphics.Rect;
import android.media.Image;
import android.os.Bundle;
import android.Manifest;
import android.content.pm.PackageManager;
import android.os.StrictMode;
import android.os.SystemClock;
import android.util.Log;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.NumberPicker;
import android.widget.PopupWindow;
import android.widget.RadioButton;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.material.switchmaterial.SwitchMaterial;
import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;

public class MainActivity extends AppCompatActivity {
    private static final String[] CAMERA_PERMISSION = new String[]{Manifest.permission.CAMERA};

    private PopupWindow popupSettings;
    private SurfaceView surfaceView;
    private TextView fpsTextView;
    private TextView modeTextView;

    private SwitchMaterial switchMonitors;
    private SwitchMaterial switchFPS;

    private NumberPicker threadsPicker;

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

    private int lensFacing = CameraSelector.LENS_FACING_FRONT;

    private BlazeFace faceDetector;
    private FaceMesh landmarksDetector;

    private Presenter presenter;

    enum DisplayMode {
        ORIGINAL,
        LANDMARKS,
        FILTERED;
        private static final DisplayMode[] vals = values();

        public DisplayMode next() {
            return vals[(this.ordinal() + 1) % vals.length];

        }
    }
    private final String LENS_FACING_KEY = "LENS_FACING";
    private final String THREADS_KEY = "THREADS_KEY";
    private final String DISPLAY_MODE_KEY = "DISPLAY_MODE";
    private final String DEVICE_KEY = "DEVICE";
    private final String MONITORS_KEY = "MONITORS";
    private final String FPS_KEY = "FPS";

    private final float IMAGE_ASPECT_RATIO = 16 / 9.f;
    private static final int CAMERA_REQUEST_CODE = 10;
    private DisplayMode displayMode = DisplayMode.FILTERED;
    private String device = "CPU";
    private int threadsNum = 4;
    private static final String TAG = "MainActivity";
    public static final String OPENCV_LIBRARY_NAME = "opencv_java4";
    public static final String FILTERS_LIBRARY_NAME = "filters";
    public static final String PRESENTER_LIBRARY_NAME = "presenter";

    private boolean showMonitors = false;
    private AtomicBoolean changeDevice = new AtomicBoolean(false);
    private AtomicBoolean changeThreadsNum = new AtomicBoolean(false);

    @Override
    public void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        outState.putInt(LENS_FACING_KEY, lensFacing);
        outState.putInt(THREADS_KEY, threadsNum);
        outState.putString(DEVICE_KEY, device);
        outState.putBoolean(MONITORS_KEY, showMonitors);
        outState.putBoolean(FPS_KEY, switchFPS.isChecked());
        outState.putSerializable(DISPLAY_MODE_KEY, displayMode);

        // Call superclass to save any view hierarchy.
        super.onSaveInstanceState(outState);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        createPopupSettingsWindow();
//        StrictMode.enableDefaults();
        // Recover the instance state.
        if (savedInstanceState != null) {
            lensFacing = savedInstanceState.getInt(LENS_FACING_KEY);
            threadsNum = savedInstanceState.getInt(THREADS_KEY);
            device = savedInstanceState.getString(DEVICE_KEY);
            showMonitors = savedInstanceState.getBoolean(MONITORS_KEY);
            switchMonitors.setChecked(savedInstanceState.getBoolean(MONITORS_KEY));
            switchFPS.setChecked(savedInstanceState.getBoolean(FPS_KEY));
            displayMode = (DisplayMode) savedInstanceState.getSerializable(DISPLAY_MODE_KEY);
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

        presenter = new Presenter("cdm", 40);
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

        modeTextView = findViewById(R.id.mode_text);
        modeTextView.setText(displayMode.name());
        Button changeDisplayMode = findViewById(R.id.changeDisplayMode);
        changeDisplayMode.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                displayMode = displayMode.next();
                modeTextView.setText(displayMode.name());
            }
        });

        ImageButton openSettings = findViewById(R.id.openSettings);
        openSettings.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                showPopupSettingsWindow();
            }
        });

        fpsTextView = findViewById(R.id.fps_text);
    }

    @Override
    public void onResume() {
        super.onResume();
        // Request permissions each time the app resumes, since they can be revoked at any time
        if (!hasCameraPermission()) {
            requestCameraPermission();
        } else {
            startCamera();
        }
    }

    @Override
    public void onPause() {
        try {
            ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
            cameraProvider.unbindAll();
        } catch (ExecutionException | InterruptedException e) {
            e.printStackTrace();
        }
        super.onPause();
    }

    @Override
    public void onDestroy() {
        try {
            ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
            cameraProvider.unbindAll();
        } catch (ExecutionException | InterruptedException e) {
            e.printStackTrace();
        }
        super.onDestroy();
    }
    private void createPopupSettingsWindow() {
        // inflate the layout of the popup window
        LayoutInflater inflater = (LayoutInflater)
                getSystemService(LAYOUT_INFLATER_SERVICE);
        View popupView = inflater.inflate(R.layout.popup_settings, null);
        popupSettings = new PopupWindow(popupView, LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT, true);

        // dismiss the popup window when touched
        popupView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                popupSettings.dismiss();
                return true;
            }
        });

        switchMonitors = popupView.findViewById(R.id.monitors_switch);
        switchMonitors.setOnCheckedChangeListener((buttonView, isChecked) -> {
            showMonitors = isChecked;
        });

        switchFPS = popupView.findViewById(R.id.fps_switch);
        switchFPS.setChecked(true);
        switchFPS.setOnCheckedChangeListener((buttonView, showFPS) -> {
            if (showFPS) {
                fpsTextView.setVisibility(View.VISIBLE);
            } else {
                fpsTextView.setVisibility(View.INVISIBLE);
            }
        });

        threadsPicker = popupView.findViewById(R.id.threads_picker);
        threadsPicker.setMinValue(1);
        threadsPicker.setMaxValue(8);
        threadsPicker.setValue(4);
        threadsPicker.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
            @Override
            public void onValueChange(NumberPicker numberPicker, int i, int i1) {
                threadsNum = numberPicker.getValue();
                changeThreadsNum.set(true);
            }
        });
    }
    private void showPopupSettingsWindow() {
        popupSettings.showAtLocation(findViewById(android.R.id.content).getRootView(), Gravity.CENTER, 0, 0);
    }

    public void onRadioButtonClicked(View view) {
        boolean checked = ((RadioButton) view).isChecked();
        switch(view.getId()) {
            case R.id.radio_cpu:
                if (checked)
                    device = "CPU";
                    break;
            case R.id.radio_gpu:
                if (checked)
                    device = "GPU";
                    break;
        }
        changeDevice.set(true);
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
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
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
        Log.d(TAG, "Init models");
        try {
            modelFile = getResourcePath(getAssets().open("face_detection_short_range.tflite"), "face_detection_short_range", "tflite");
        } catch (IOException ex) {
        }
        faceDetector = new BlazeFace(modelFile, device, threadsNum);

        try {
            modelFile = getResourcePath(getAssets().open("face_landmark.tflite"), "face_landmark", "tflite");
        } catch (IOException ex) {
        }
        landmarksDetector = new FaceMesh(modelFile, device, threadsNum);
    }

    private void drawImage(Bitmap bitmap) {
        Rect src = new Rect(0, 0, bitmap.getWidth(), bitmap.getHeight());
        Rect dst;
        int w = surfaceView.getWidth();
        int h = surfaceView.getHeight();
        if (w < h) {
            h = Math.round(w * IMAGE_ASPECT_RATIO);
        } else {
            h = Math.round(w / IMAGE_ASPECT_RATIO);
        }
        dst = new Rect(0, 0, w, h);
        SurfaceHolder holder = surfaceView.getHolder();
        Canvas canvas = holder.lockCanvas();
        canvas.drawBitmap(bitmap, src, dst, null);
        holder.unlockCanvasAndPost(canvas);
    }

    private static Bitmap toBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer buffer = planes[0].getBuffer();
        int pixelStride = planes[0].getPixelStride();
        int rowStride = planes[0].getRowStride();
        int rowPadding = rowStride - pixelStride * image.getWidth();
        Bitmap bitmap = Bitmap.createBitmap(image.getWidth() + rowPadding/pixelStride,
                image.getHeight(), Bitmap.Config.ARGB_8888);
        bitmap.copyPixelsFromBuffer(buffer);
        return bitmap;
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

    private void bindImageAnalysis(@NonNull ProcessCameraProvider cameraProvider) {
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
                long startTime = SystemClock.elapsedRealtime();
                if ((faceDetector == null) || (landmarksDetector == null)
                        || changeDevice.compareAndSet(true, false)
                        || (changeThreadsNum.compareAndSet(true, false) && device == "CPU")) {
                    initModels();
                }
                int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();
                Bitmap bmp = toBitmap(imageProxy.getImage());
                Matrix matrix = new Matrix();
                float scaleW =  854.f / bmp.getWidth();
                float scaleH = 480.f / bmp.getHeight();
                matrix.postRotate((float)rotationDegrees);
                if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                    // to keep image upright on camera rotation
                    matrix.postScale(-scaleW, scaleH);
                } else {
                    matrix.postScale(scaleW, scaleH);
                }

                bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true);
                Mat frame = new Mat();
                Utils.bitmapToMat(bmp, frame);
                Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
                Utils.matToBitmap(frame, bmp);
                ArrayList<BBox> boxes = faceDetector.run(bmp, null);
                ArrayList<FacialLandmarks> lms = new ArrayList<>();

                for (BBox b : boxes) {
                    FaceMeshMData mdata = new FaceMeshMData(b.face, b.leftEye, b.rightEye);
                    lms.add(landmarksDetector.run(bmp, mdata));
                }

                if (!lms.isEmpty() && displayMode != DisplayMode.ORIGINAL) {
                    if (displayMode == DisplayMode.FILTERED)
                        Filters.beautifyFace(frame, lms);
                    if (displayMode == DisplayMode.LANDMARKS)
                        renderResults(frame, lms);
                }

                double fps = 1000.f / (SystemClock.elapsedRealtime() - startTime);
                fpsTextView.setText(String.format("%.2f", fps) + " FPS");

                if (showMonitors) {
                    presenter.drawGraphs(frame);
                }

                Utils.matToBitmap(frame, bmp);
                drawImage(bmp);
                imageProxy.close();
            }
        });

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(lensFacing).build();

        try {
            cameraProvider.unbindAll();
            cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalysis);
        } catch (Exception ex) {
            Log.e(TAG, "Failed to bind use case");
        }
    }
}
