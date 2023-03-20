package com.example.test;

import android.graphics.Bitmap;
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

    public FaceMesh(final String modelFile, final int nthreads) {
        super(modelFile, nthreads);
        getInputsOutputsInfo();
        imgProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeLetterbox(inputHeight, inputWidth))
                        .add(new NormalizeOp(new float[]{0.f,0.f,0.f}, scale))
                        .build();

        landmarksBuffer = TensorBuffer.createFixedSize(outputShapes.get(0), outputDataTypes.get(0));
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
    @Override
    protected final TensorImage preprocess(Bitmap bitmap) {
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(bitmap);
        tensorImage = imgProcessor.process(tensorImage);
        return tensorImage;
    }
    @Override
    protected ArrayList<FacialLandmarks> postprocess() {
        float[] landmarks = tensorProcessor.process(landmarksBuffer).getFloatArray();

        ArrayList<FacialLandmarks> lms = new ArrayList<>();

        return lms;
    }
    @Override
    public final ArrayList<FacialLandmarks> run(Bitmap bitmap, MetaData mdata) {
        imageWidth = bitmap.getWidth();
        imageHeight = bitmap.getHeight();
        TensorImage tensorImage = preprocess(bitmap);
        interpreter.run(tensorImage.getBuffer(), landmarksBuffer.getBuffer().rewind());
        return postprocess();
    }
}
