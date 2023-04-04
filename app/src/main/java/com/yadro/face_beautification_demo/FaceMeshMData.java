package com.yadro.face_beautification_demo;

import android.graphics.PointF;
import android.graphics.Rect;

public class FaceMeshMData extends MetaData {
    Rect faceRect;
    PointF leftEye;
    PointF rightEye;

    public FaceMeshMData(Rect faceRect, PointF leftEye, PointF rightEye) {
        this.faceRect = faceRect;
        this.leftEye = leftEye;
        this.rightEye = rightEye;
    }
}
