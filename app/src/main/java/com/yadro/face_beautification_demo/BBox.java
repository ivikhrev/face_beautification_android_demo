package com.yadro.face_beautification_demo;

import android.graphics.PointF;
import android.graphics.Rect;

public class BBox {
    Rect face;
//    float left;
//    float top;
//    float right;
//    float bottom;

    PointF leftEye;
    PointF rightEye;
    PointF nose;
    PointF mouth;
    PointF leftTragion;
    PointF rightTragion;

    float confidence;
}
