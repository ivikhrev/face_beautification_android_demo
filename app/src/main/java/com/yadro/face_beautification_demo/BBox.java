package com.yadro.face_beautification_demo;

import android.graphics.PointF;
import android.graphics.Rect;

public class BBox {
    Rect face;
    PointF leftEye;
    PointF rightEye;
    PointF nose;
    PointF mouth;
    PointF leftTragion;
    PointF rightTragion;

    float confidence;
}
