package com.example.test;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.PointF;
import android.graphics.Rect;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class FaceMesh extends TFLiteModel<FacialLandmarks> {
    private TensorBuffer landmarksBuffer;
    private TensorBuffer scoresBuffer;
    private final TensorProcessor tensorProcessor;

    static final private float roiEnlargeCoeff = 1.5f;
    
    int[] faceOvalIdx = {10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109};

    int[] leftEyeIdx = {130, 7, 163, 144, 145, 153, 154, 155, 133, 173, 56, 28, 27, 29, 30, 247};

    int[] leftBrowIdx = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46};

    int[] rightEyeIdx =
            {362, 382, 381, 380, 374, 373, 373, 390, 249, 359, 467, 260, 259, 257, 258, 286, 414, 463};

    int[] rightBrowIdx = {336, 296, 334, 293, 300, 276, 283, 282, 295, 285};

    int[] noseIdx = {2, 99, 240, 235, 219, 218, 237, 44, 19, 274, 457, 438, 392, 289, 305, 328};

    int[] lipsIdx = {61,  146, 91,  181, 84,  17, 314, 405, 321, 375,
            291, 409, 270, 269, 267, 0,  37,  39,  40,  185};
    private float[] scale = {255.0f, 255.0f, 255.0f};

    private PointF rotationCenter;
    private double rotationRad;

    private int faceRoiWidth;
    private int faceRoiHeight;


    public FaceMesh(final String modelFile, final int nthreads) {
        super(modelFile, nthreads);
        getInputsOutputsInfo();
        imgProcessor =
                new ImageProcessor.Builder()
                        .add(new NormalizeOp(new float[]{0.f,0.f,0.f}, scale))
                        .build();

        landmarksBuffer = TensorBuffer.createFixedSize(outputShapes.get(0), outputDataTypes.get(0));
        scoresBuffer = TensorBuffer.createFixedSize(outputShapes.get(1), outputDataTypes.get(1));
        TensorProcessor.Builder builder = new TensorProcessor.Builder();
        tensorProcessor = builder.build();
    }
    @Override
    protected void getInputsOutputsInfo() {
        int inputsCount = interpreter.getInputTensorCount();
        Log.i(TAG, "Inputs:");
        Tensor tensor = interpreter.getInputTensor(0);
        Log.i(TAG, "\t" + tensor.name() + ": " + String.valueOf(tensor.dataType()) + " " + Arrays.toString(tensor.shape()));
        inputWidth = tensor.shape()[2];
        inputHeight = tensor.shape()[1];

        int outputsCount = interpreter.getOutputTensorCount();
        Log.i(TAG, "Outputs: ");
        for (int i = 0; i < outputsCount; ++i) {
            tensor = interpreter.getOutputTensor(i);
            outputNames.add(tensor.name());
            outputDataTypes.add(tensor.dataType());
            outputShapes.add(tensor.shape());
            Log.i(TAG, "\t" + tensor.name() + ": " + String.valueOf(tensor.dataType()) + " " + Arrays.toString(tensor.shape()));
        }
    }

    public static final  Rect enlargeFaceRoi(Rect roi, int width, int height) {
        Rect enlargedRoi = new Rect();
        int inflationX = (Math.round(roi.width() * roiEnlargeCoeff) - roi.width()) / 2;
        int inflationY = (Math.round(roi.height() * roiEnlargeCoeff) - roi.height()) / 2;
        enlargedRoi.left = (roi.left - inflationX) < 0 ? 0 : roi.left - inflationX;
        enlargedRoi.top =  (roi.top - inflationY) < 0 ? 0 : roi.top - inflationY;
        enlargedRoi.right = (roi.right + inflationX) > width ? width : roi.right + inflationX;
        enlargedRoi.bottom = (roi.bottom + inflationY) > height ? height : roi.bottom + inflationY;
        return enlargedRoi;
    }
    public static final double calculateRotationRad(PointF p0, PointF p1) {
        double rad = -Math.atan2(p0.y - p1.y, p1.x - p0.x);
        double radNormed = rad - 2 * Math.PI * Math.floor((rad + Math.PI) / (2 * Math.PI));  // normalized to [0, 2*PI]
        return radNormed;
    }

    public static final float[] rotatePoints(float[] points,
                                   double rad,
                                   final PointF rotCenter) {
        double sin = Math.sin(rad);
        double cos = Math.cos(rad);
        float[] res = new float[points.length];
        for (int i = 0; i < points.length / 2; ++i) {
            float x = points[2 * i];
            float y = points[2 * i + 1];
            x -= rotCenter.x;
            y -= rotCenter.y;
            float newX = (float) (x * cos - y * sin);
            float newY = (float) (x * sin + y * cos);
            newX += rotCenter.x;
            newY += rotCenter.y;

            res[2 * i] = newX;
            res[2 * i + 1] = newY;
        }
        return res;
    }

    @Override
    protected final TensorImage preprocess(Bitmap bitmap) {
        FaceMeshMData data = (FaceMeshMData) mdata;
        Rect faceRect = enlargeFaceRoi(data.faceRect, imageWidth, imageHeight);
        faceRoiWidth = faceRect.width();
        faceRoiHeight = faceRect.height();
        rotationCenter = new PointF((faceRect.left + faceRect.right) * 0.5f, (faceRect.top + faceRect.bottom) * 0.5f);
        rotationRad = calculateRotationRad(data.leftEye, data.rightEye);

        float[] dstPoints = {0, 0,
                inputWidth, 0,
                inputWidth, inputHeight,
                0, inputHeight};
        float[] srcPoints = {faceRect.left, faceRect.top,
                faceRect.right, faceRect.top,
                faceRect.right, faceRect.bottom,
                faceRect.left, faceRect.bottom};

        srcPoints = rotatePoints(srcPoints, rotationRad, rotationCenter);

        Matrix m = new Matrix();
        m.setPolyToPoly(srcPoints, 0, dstPoints, 0, dstPoints.length >> 1);
        Bitmap dstBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(dstBitmap);
        canvas.clipRect(0, 0, inputWidth, inputHeight);
        canvas.drawBitmap(bitmap, m, null);

        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(dstBitmap);
        tensorImage = imgProcessor.process(tensorImage);
        return tensorImage;
    }

    public static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }
    protected ArrayList<PointF> fillLandmarks(float[] landmarks, int[] ids) {
        ArrayList<PointF> res = new ArrayList<>();
        for (int i : ids) {
            final int numAxis = 3;
            float x = (landmarks[i * numAxis]) / inputWidth;
            float y = (landmarks[i * numAxis + 1]) / inputHeight;
            // rotate
            float[] rotPoint = rotatePoints(new float[]{x, y}, rotationRad, new PointF(0.5f, 0.5f));
            // map back to img coordinates
            float offsetX = rotationCenter.x - faceRoiWidth / 2.f;
            float offsetY = rotationCenter.y - faceRoiHeight / 2.f;
            rotPoint = new float[]{rotPoint[0] * faceRoiWidth + offsetX, rotPoint[1] * faceRoiHeight + offsetY};
            res.add(new PointF(clamp(rotPoint[0], 0,imageWidth), clamp(rotPoint[1], 0, imageHeight)));
        }
        return res;
    }
    @Override
    protected FacialLandmarks postprocess() {
        float[] landmarks = tensorProcessor.process(landmarksBuffer).getFloatArray();
        float[] scores = tensorProcessor.process(scoresBuffer).getFloatArray();
        FacialLandmarks lms = new FacialLandmarks();

        lms.faceOval = fillLandmarks(landmarks, faceOvalIdx);
        lms.leftEye = fillLandmarks(landmarks, leftEyeIdx);
        lms.leftBrow = fillLandmarks(landmarks, leftBrowIdx);
        lms.rightEye = fillLandmarks(landmarks, rightEyeIdx);
        lms.rightBrow = fillLandmarks(landmarks, rightBrowIdx);
        lms.nose = fillLandmarks(landmarks, noseIdx);
        lms.lips = fillLandmarks(landmarks, lipsIdx);

        return lms;
    }

    @Override
    public final FacialLandmarks run(Bitmap bitmap, MetaData mdata) {
        imageWidth = bitmap.getWidth();
        imageHeight = bitmap.getHeight();
        this.mdata = mdata;
        TensorImage tensorImage = preprocess(bitmap);
        interpreter.run(tensorImage.getBuffer(), landmarksBuffer.getBuffer().rewind());
        return postprocess();
    }
}
