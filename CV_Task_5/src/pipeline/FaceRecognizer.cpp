#include "FaceRecognizer.h"

#include "../core/MathEngine.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace facerecog {

namespace {

struct Neighbor {
    int identity = -1;
    std::string label;
    double distance = 0.0;
};

struct VoteBucket {
    int identity = -1;
    std::string label;
    int votes = 0;
    double distanceSum = 0.0;
};

} // namespace

FaceRecognizer::FaceRecognizer(const PCAPipeline& pca)
    : m_pca(pca)
{
}

RecognitionResult FaceRecognizer::recognize(const QImage& faceImage, int k) const
{
    if (faceImage.isNull()) {
        throw std::invalid_argument("FaceRecognizer::recognize - input image is null.");
    }

    return recognize(m_pca.imageToVector(faceImage), k);
}

RecognitionResult FaceRecognizer::recognize(const Vector& faceVector, int k) const
{
    if (!m_pca.isTrained()) {
        throw std::runtime_error("FaceRecognizer::recognize - PCA model is not trained.");
    }

    const auto& projections = m_pca.getTrainingProjections();
    if (projections.empty()) {
        throw std::runtime_error("FaceRecognizer::recognize - no training projections are available.");
    }

    k = std::max(1, std::min(k, static_cast<int>(projections.size())));

    const Vector queryWeights = m_pca.project(faceVector);

    std::vector<Neighbor> neighbors;
    neighbors.reserve(projections.size());

    for (const ProjectedFaceSample& sample : projections) {
        Neighbor n;
        n.identity = sample.identity;
        n.label = sample.label;
        n.distance = MathEngine::euclideanDistance(queryWeights, sample.weights);
        neighbors.push_back(std::move(n));
    }

    std::sort(neighbors.begin(), neighbors.end(),
              [](const Neighbor& a, const Neighbor& b) {
                  return a.distance < b.distance;
              });

    std::map<int, VoteBucket> buckets;

    for (int i = 0; i < k; ++i) {
        const Neighbor& n = neighbors[static_cast<std::size_t>(i)];
        auto& bucket = buckets[n.identity];
        bucket.identity = n.identity;
        bucket.label = n.label;
        bucket.votes += 1;
        bucket.distanceSum += n.distance;
    }

    VoteBucket best;
    best.identity = -1;
    best.votes = -1;
    best.distanceSum = std::numeric_limits<double>::infinity();

    for (const auto& item : buckets) {
        const VoteBucket& candidate = item.second;
        const double candidateAverage = candidate.distanceSum / candidate.votes;
        const double bestAverage = best.distanceSum / std::max(1, best.votes);

        if (candidate.votes > best.votes
            || (candidate.votes == best.votes && candidateAverage < bestAverage)) {
            best = candidate;
        }
    }

    RecognitionResult result;
    result.identity = best.identity;
    result.label = best.label;
    result.nearestDistance = neighbors.front().distance;
    result.averageWinningDistance = best.distanceSum / std::max(1, best.votes);

    const double scale = std::sqrt(static_cast<double>(std::max<std::size_t>(1, m_pca.getK()))) * 255.0;
    result.confidence = 1.0 / (1.0 + result.averageWinningDistance / std::max(1.0, scale));

    for (int i = 0; i < k; ++i) {
        result.nearestNeighbors.emplace_back(neighbors[static_cast<std::size_t>(i)].label,
                                             neighbors[static_cast<std::size_t>(i)].distance);
    }

    return result;
}

} // namespace facerecog
