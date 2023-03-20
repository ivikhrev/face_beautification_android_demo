//
// Created by i.vikhrev on 3/17/2023.
//

#include <jni.h>
#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <numeric>

struct {
    int numLayers = 4;
    int numBoxes = 896;
    int numPoints = 16;
    float anchorOffsetX = 0.5;
    float anchorOffsetY = 0.5;
    std::vector<int> strides = {8, 16, 16, 16};
    double interpolatedScaleAspectRatio = 1.0;
} ssdModelOptions;

struct BBox {
    float left;
    float top;
    float right;
    float bottom;

    std::array<float, 2> leftEye;
    std::array<float, 2> rightEye;
    std::array<float, 2> nose;
    std::array<float, 2> mouth;
    std::array<float, 2> leftTragion;
    std::array<float, 2> rightTragion;

    float confidence;
};

template <typename Anchor>
std::vector<int> nms(const std::vector<Anchor>& boxes, const std::vector<float>& scores,
                     const float thresh, bool includeBoundaries=false) {
    std::vector<float> areas(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        areas[i] = (boxes[i].right - boxes[i].left + includeBoundaries) * (boxes[i].bottom - boxes[i].top + includeBoundaries);
    }
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&scores](int o1, int o2) { return scores[o1] > scores[o2]; });

    size_t ordersNum = 0;
    for (; ordersNum < order.size() && scores[order[ordersNum]] >= 0; ordersNum++);

    std::vector<int> keep;
    bool shouldContinue = true;
    for (size_t i = 0; shouldContinue && i < ordersNum; ++i) {
        auto idx1 = order[i];
        if (idx1 >= 0) {
            keep.push_back(idx1);
            shouldContinue = false;
            for (size_t j = i + 1; j < ordersNum; ++j) {
                auto idx2 = order[j];
                if (idx2 >= 0) {
                    shouldContinue = true;
                    auto overlappingWidth = std::fminf(boxes[idx1].right, boxes[idx2].right) - std::fmaxf(boxes[idx1].left, boxes[idx2].left);
                    auto overlappingHeight = std::fminf(boxes[idx1].bottom, boxes[idx2].bottom) - std::fmaxf(boxes[idx1].top, boxes[idx2].top);
                    auto intersection = overlappingWidth > 0 && overlappingHeight > 0 ? overlappingWidth * overlappingHeight : 0;
                    auto overlap = intersection / (areas[idx1] + areas[idx2] - intersection);

                    if (overlap >= thresh) {
                        order[j] = -1;
                    }
                }
            }
        }
    }
    return keep;
}

void generateAnchors() {
    int layerId = 0;
    while (layerId < ssdModelOptions.numLayers) {
        int lastSameStrideLayer = layerId;
        int repeats = 0;
        while (lastSameStrideLayer < ssdModelOptions.numLayers &&
               ssdModelOptions.strides[lastSameStrideLayer] == ssdModelOptions.strides[layerId]) {
            lastSameStrideLayer += 1;
            repeats += 2;
        }
        size_t stride = ssdModelOptions.strides[layerId];
        int featureMapHeight = inputHeight / static_cast<int>(stride);
        int featureMapWidth = inputWidth / static_cast<int>(stride);
        for (int y = 0; y < featureMapHeight; ++y) {
            float yCenter =
                    (static_cast<float>(y) + ssdModelOptions.anchorOffsetY) / static_cast<float>(featureMapHeight);
            for (int x = 0; x < featureMapWidth; ++x) {
                float xCenter =
                        (static_cast<float>(x) + ssdModelOptions.anchorOffsetX) / static_cast<float>(featureMapWidth);
                for (int i = 0; i < repeats; ++i) {
                    anchors.emplace_back(xCenter, yCenter);
                }
            }
        }

        layerId = lastSameStrideLayer;
    }
}

void BlazeFace::decodeBox(float* boxes, size_t boxId) {
    const float scale = static_cast<float>(inputHeight);
    const size_t numPoints = ssdModelOptions.numPoints / 2;
    const int startPos = boxId * static_cast<int>(ssdModelOptions.numPoints);

    for (size_t j = 0; j < numPoints; ++j) {
        boxes[startPos + 2 * j] = boxes[startPos + 2 * j] / scale;
        boxes[startPos + 2 * j + 1] = boxes[startPos + 2 * j + 1] / scale;
        if (j != 1) {
            boxes[startPos + 2 * j] += anchors[boxId].x;
            boxes[startPos + 2 * j + 1] += anchors[boxId].y;
        }
    }

    // convert x center, y center, w, h to xmin, ymin, xmax, ymax
    const float halfWidth = boxes[startPos + 2] / 2;
    const float halfHeight = boxes[startPos + 3] / 2;
    const float xCenter = boxes[startPos];
    const float yCenter = boxes[startPos + 1];

    boxes[startPos] -= halfWidth;
    boxes[startPos + 1] -= halfHeight;

    boxes[startPos + 2] = xCenter + halfWidth;
    boxes[startPos + 3] = yCenter + halfHeight;
}
}

std::pair<std::vector<BBox>, std::vector<float>> BlazeFace::getDetections(const float* scores, float* boxes) {
    std::vector<BBox> detections;
    std::vector<float> filteredScores;
    for (int boxId = 0; boxId < ssdModelOptions.numBoxes; ++boxId) {
        float score = scores[boxId];
        if (score < logitThreshold) {
            continue;
        }

        BBox object;
        object.confidence = 1.f / (1.f + std::exp(-score));

        decodeBox(boxes, boxId);
        const int startPos = boxId * ssdModelOptions.numPoints;
        // unscale from letterbox
        const float x0 = (boxes[startPos] - xPadding) / xScale;
        const float y0 = (boxes[startPos + 1] - yPadding) / yScale;
        const float x1 = (boxes[startPos + 2] - xPadding) / xScale;
        const float y1 = (boxes[startPos + 3] - yPadding) / yScale;

        const float xLeftEye = (boxes[startPos + 4] - xPadding) / xScale;
        const float yLeftEye = (boxes[startPos + 5] - yPadding) / yScale;

        const float xRightEye = (boxes[startPos + 6] - xPadding) / xScale;
        const float yRightEye = (boxes[startPos + 7] - yPadding) / yScale;

        const float xNose = (boxes[startPos + 8] - xPadding) / xScale;
        const float yNose = (boxes[startPos + 9] - yPadding) / yScale;

        const float xMouth = (boxes[startPos + 10] - xPadding) / xScale;
        const float yMouth = (boxes[startPos + 11] - yPadding) / yScale;

        const float xLeftTragion = (boxes[startPos + 12] - xPadding) / xScale;
        const float yLeftTragion = (boxes[startPos + 13] - yPadding) / yScale;

        const float xRightTragion = (boxes[startPos + 14] - xPadding) / xScale;
        const float yRightTragion = (boxes[startPos + 15] - yPadding) / yScale;

        object.left = std::clamp(x0 * origImageWidth, 0.f, static_cast<float>(origImageWidth));
        object.top = std::clamp(y0 * origImageHeight, 0.f, static_cast<float>(origImageHeight));
        object.right = std::clamp(x1 * origImageWidth, 0.f, static_cast<float>(origImageWidth));
        object.bottom = std::clamp(y1 * origImageHeight, 0.f, static_cast<float>(origImageHeight));

        object.leftEye = {std::clamp(xLeftEye * origImageWidth, 0.f, static_cast<float>(origImageWidth)),
                          std::clamp(yLeftEye * origImageHeight, 0.f, static_cast<float>(origImageHeight))};
        object.rightEye = {std::clamp(xRightEye * origImageWidth, 0.f, static_cast<float>(origImageWidth)),
                           std::clamp(yRightEye * origImageHeight, 0.f, static_cast<float>(origImageHeight))};
        object.nose = {std::clamp(xNose * origImageWidth, 0.f, static_cast<float>(origImageWidth)),
                       std::clamp(yNose * origImageHeight, 0.f, static_cast<float>(origImageHeight))};
        object.mouth = {std::clamp(xMouth * origImageWidth, 0.f, static_cast<float>(origImageWidth)),
                        std::clamp(yMouth * origImageHeight, 0.f, static_cast<float>(origImageHeight))};
        object.leftTragion = {std::clamp(xLeftTragion * origImageWidth, 0.f, static_cast<float>(origImageWidth)),
                              std::clamp(yLeftTragion * origImageHeight, 0.f, static_cast<float>(origImageHeight))};
        object.rightTragion = {std::clamp(xRightTragion * origImageWidth, 0.f, static_cast<float>(origImageWidth)),
                               std::clamp(yRightTragion * origImageHeight, 0.f, static_cast<float>(origImageHeight))};

        filteredScores.push_back(score);
        detections.push_back(object);
    }

    return {detections, filteredScores};
}


std::unique_ptr<Result> postprocess() {
    std::vector<BBox> faces;
    float* boxesPtr = interpreter->typed_output_tensor<float>(boxesTensorId);
    float* scoresPtr = interpreter->typed_output_tensor<float>(scoresTensorId);

    auto [detections, filteredScores] = getDetections(scoresPtr, boxesPtr);
    std::vector<int> keep = nms(detections, filteredScores, 0.5);
    DetectionResult* result = new DetectionResult();
    for (auto& index : keep) {
        result->boxes.push_back(detections[index]);
    }
    return std::unique_ptr<Result>(result);
}
}
