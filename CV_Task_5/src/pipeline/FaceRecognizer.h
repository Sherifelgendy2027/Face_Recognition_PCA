#ifndef FACERECOGNIZER_H
#define FACERECOGNIZER_H

/**
 * @file FaceRecognizer.h
 * @brief Step 4: Native K-nearest-neighbors recognition in PCA face space.
 */

#include "PCAPipeline.h"

#include <QImage>

#include <string>
#include <vector>

namespace facerecog {

struct RecognitionResult {
    int identity = -1;
    std::string label;
    double nearestDistance = 0.0;
    double averageWinningDistance = 0.0;
    double confidence = 0.0;
    std::vector<std::pair<std::string, double>> nearestNeighbors;
};

class FaceRecognizer {
public:
    explicit FaceRecognizer(const PCAPipeline& pca);

    RecognitionResult recognize(const QImage& faceImage, int k = 3) const;
    RecognitionResult recognize(const Vector& faceVector, int k = 3) const;

private:
    const PCAPipeline& m_pca;
};

} // namespace facerecog

#endif // FACERECOGNIZER_H
