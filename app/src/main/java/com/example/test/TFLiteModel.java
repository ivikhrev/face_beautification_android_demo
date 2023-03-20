package com.example.test;

import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

abstract public class TFLiteModel<T> {
    protected static final String TAG = "TFLiteModel";

    protected MappedByteBuffer model;
    protected Interpreter interpreter;
    protected final Interpreter.Options options = new Interpreter.Options();

    protected int nthreads;
    protected int imageWidth;
    protected int imageHeight;
    protected int inputWidth;
    protected int inputHeight;
    protected ImageProcessor imgProcessor;
    protected ArrayList<String> outputNames = new ArrayList<String>();
    protected ArrayList<DataType> outputDataTypes  = new ArrayList<DataType>();
    protected ArrayList<int[]> outputShapes  = new ArrayList<int[]>();

    protected MetaData mdata;

    private MappedByteBuffer loadModelFile(String modelPath) throws IOException {
        Log.i(TAG, "Load asset model file: " + modelPath);
        File file=new File(modelPath);
        FileInputStream inputStream = new FileInputStream(file);
        FileChannel fileChannel = inputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
    }

    public TFLiteModel(final String modelFile, final int nthreads) {
        this.nthreads = nthreads;
        try {
            model = loadModelFile(modelFile);
        } catch (IOException ex) {
            Log.e("IO Error",
                    "Failed to load model asset file" + ex.toString());
            System.exit(1);
        }
        readModel(modelFile);
    }

    protected void readModel(final String modelFile) {
        Log.i(TAG, "Reading model");
        options.setNumThreads(nthreads);
        interpreter = new Interpreter(model, options);
    }

    abstract protected void getInputsOutputsInfo();

    abstract protected TensorImage preprocess(Bitmap bitmap);
    abstract public T run(Bitmap bitmap, MetaData mdata);
    abstract protected T postprocess();

}
