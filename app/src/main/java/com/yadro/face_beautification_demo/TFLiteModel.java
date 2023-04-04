package com.yadro.face_beautification_demo;

import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;

abstract public class TFLiteModel<T> {
    protected static final String TAG = "TFLiteModel";

    protected MappedByteBuffer model;
    protected Interpreter interpreter;
    protected final Interpreter.Options options = new Interpreter.Options();
    protected final CompatibilityList compatList = new CompatibilityList();
    protected int nthreads;
    protected int imageWidth;
    protected int imageHeight;
    protected int inputWidth;
    protected int inputHeight;
    protected ImageProcessor imgProcessor;
    protected ArrayList<String> outputNames = new ArrayList<>();
    protected ArrayList<DataType> outputDataTypes  = new ArrayList<>();
    protected ArrayList<int[]> outputShapes  = new ArrayList<>();

    protected MetaData mdata;

    protected String device = "CPU";

    private MappedByteBuffer loadModelFile(String modelPath) throws IOException {
        Log.i(TAG, "Load asset model file: " + modelPath);
        File file=new File(modelPath);
        FileInputStream inputStream = new FileInputStream(file);
        FileChannel fileChannel = inputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
    }

    public TFLiteModel(final String modelFile, final String device, final int nthreads) {
        this.nthreads = nthreads;
        this.device = device;
        try {
            model = loadModelFile(modelFile);
        } catch (IOException ex) {
            Log.e("IO Error",
                    "Failed to load model asset file" + ex.toString());
            System.exit(1);
        }

        readModel(modelFile);
    }

    protected void readModel(final String modelFile) throws IllegalArgumentException {
        Log.i(TAG, "Reading model");
//        boolean isGPUSupported = compatList.isDelegateSupportedOnThisDevice();
        if(device.equals("GPU")) {
            // if the device has a supported GPU, add the GPU delegate
            GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            options.addDelegate(gpuDelegate);
            Log.i(TAG, "Delegate: GPU");
//        } else if (device.equals("GPU") && !isGPUSupported) {
//            Log.w(TAG, "GPU Delegate is unsupported on this device! Fallback to CPU");
//            options.setNumThreads(nthreads);
//            options.setUseXNNPACK(true);
//            Log.i(TAG, "Delegate: CPU");
//            Log.i(TAG, "Threads number: " + nthreads);
        } else if (device.equals("CPU")) {
            options.setNumThreads(nthreads);
            options.setUseXNNPACK(true);
            Log.i(TAG, "Delegate: CPU");
            Log.i(TAG, "Threads number: " + nthreads);
        }
        else {
            throw new IllegalArgumentException("Unknown device provided: " + device);
        }
        interpreter = new Interpreter(model, options);
    }

    abstract protected void getInputsOutputsInfo();

    abstract protected TensorImage preprocess(Bitmap bitmap);
    abstract public T run(Bitmap bitmap, MetaData mdata);
    abstract protected T postprocess();

}
