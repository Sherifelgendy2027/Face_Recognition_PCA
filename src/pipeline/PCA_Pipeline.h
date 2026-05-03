#ifndef PCA_PIPELINE_H
#define PCA_PIPELINE_H

/**
 * @file PCA_Pipeline.h
 * @brief PCA (Eigenfaces) training pipeline for face detection and recognition.
 *
 * Handles dataset loading (AT&T/ORL format with numeric subfolders),
 * grayscale conversion, image flattening, PCA training via the
 * Turk-Pentland trick, and projection/reconstruction in eigenface space.
 *
 * Dataset expected format:
 *   dataset_root/
 *     1/   ← person 1
 *       1.pgm, 2.pgm, ...
 *     2/   ← person 2
 *       1.pgm, 2.pgm, ...
 *     ...
 *     41/  ← person 41
 */

#include "core/MathEngine.h"

#include <string>
#include <vector>
#include <cstddef>

// Forward-declare Qt types to avoid pulling in headers
class QImage;

namespace facerecog {

/**
 * @brief A single training sample: flattened pixel data + identity label.
 */
struct FaceSample {
    Vector pixels;         ///< Flattened grayscale pixels (D = width * height)
    std::string label;     ///< Identity label (subfolder name, e.g., "1", "2")
    std::string filePath;  ///< Original file path for debugging
};

/**
 * @brief Result returned after PCA training.
 */
struct TrainResult {
    size_t numSamples;        ///< Total training images loaded
    size_t numClasses;        ///< Number of distinct identities
    size_t dimension;         ///< Pixel dimension D = width × height
    size_t selectedK;         ///< Number of eigenfaces selected
    double varianceCaptured;  ///< Actual cumulative variance ratio captured by K
};

/**
 * @brief PCA pipeline: load → train → project → reconstruct.
 */
class PCA_Pipeline {
public:
    /// Standard face dimensions (AT&T/ORL: 92 × 112)
    static constexpr int FACE_WIDTH  = 92;
    static constexpr int FACE_HEIGHT = 112;
    static constexpr int FACE_DIM    = FACE_WIDTH * FACE_HEIGHT; // 10304

    PCA_Pipeline();

    // ── Dataset Loading ────────────────────────────────────────────────────

    /**
     * @brief Load all face images from a dataset directory.
     *
     * Expects: rootPath / {1,2,...,N} / {image files}
     * Each image is converted to grayscale, resized to FACE_WIDTH × FACE_HEIGHT,
     * and flattened to a 1D vector of doubles in [0, 255].
     *
     * @param rootPath Path to the dataset root folder
     * @return Number of images loaded
     * @throws std::runtime_error if directory cannot be opened or no images found
     */
    size_t loadDataset(const std::string& rootPath);

    /**
     * @brief Get all loaded samples.
     */
    const std::vector<FaceSample>& getSamples() const { return m_samples; }

    /**
     * @brief Get the number of distinct identity labels.
     */
    size_t getNumClasses() const;

    // ── Training ───────────────────────────────────────────────────────────

    /**
     * @brief Train the PCA model on loaded samples.
     *
     * Steps:
     *   1. Build data matrix (N × D) from flattened samples
     *   2. Compute mean face μ
     *   3. Center data: A = X − μ
     *   4. Compute small covariance (Turk-Pentland): C = A · Aᵀ  (N × N)
     *   5. Eigendecompose C → eigenvalues, small eigenvectors
     *   6. Recover full eigenvectors: uᵢ = Aᵀ · vᵢ / ‖Aᵀ · vᵢ‖
     *   7. Auto-select K for targetVariance (default 95%)
     *   8. Project all training faces into K-dim subspace
     *
     * @param targetVariance Fraction of variance to capture (default 0.95)
     * @return TrainResult with summary statistics
     * @throws std::runtime_error if no samples loaded
     */
    TrainResult train(double targetVariance = 0.95);

    /**
     * @brief Check whether the model has been trained.
     */
    bool isTrained() const { return m_trained; }

    // ── Projection & Reconstruction ────────────────────────────────────────

    /**
     * @brief Project a face vector into eigenface space.
     *
     * Computes: w = Eᵀ · (x − μ)
     * where E is the K × D eigenface matrix.
     *
     * @param face Flattened face vector of dimension D
     * @return Weight vector of dimension K
     */
    Vector project(const Vector& face) const;

    /**
     * @brief Reconstruct a face from its eigenface weights.
     *
     * Computes: x̂ = μ + E · w (using only top K eigenfaces)
     *
     * @param weights Weight vector of dimension K
     * @return Reconstructed face vector of dimension D
     */
    Vector reconstruct(const Vector& weights) const;

    /**
     * @brief Compute reconstruction error for a face vector.
     *
     * ‖x − x̂‖² where x̂ = reconstruct(project(x))
     *
     * @param face Flattened face vector of dimension D
     * @return Squared reconstruction error
     */
    double reconstructionError(const Vector& face) const;

    // ── Accessors ──────────────────────────────────────────────────────────

    const Vector& getMeanFace() const { return m_meanFace; }
    const Matrix& getEigenfaces() const { return m_eigenfaces; }
    const Vector& getEigenvalues() const { return m_eigenvalues; }
    size_t getK() const { return m_K; }

    /**
     * @brief Get the projected coordinates and labels of all training faces.
     * @return Pairs of (weight vector, label)
     */
    const std::vector<std::pair<Vector, std::string>>& getTrainingProjections() const
    {
        return m_trainingProjections;
    }

    // ── Image Utilities ────────────────────────────────────────────────────

    /**
     * @brief Convert a QImage to a grayscale, resized, flattened vector.
     * @param image Input QImage (any format)
     * @return Flattened vector of dimension FACE_DIM with values in [0, 255]
     */
    static Vector imageToVector(const QImage& image);

    /**
     * @brief Convert a flattened pixel vector back to a QImage (grayscale).
     * @param pixels Flattened vector of dimension FACE_DIM
     * @return Grayscale QImage of size FACE_WIDTH × FACE_HEIGHT
     */
    static QImage vectorToImage(const Vector& pixels);

private:
    // ── Data ───────────────────────────────────────────────────────────────
    std::vector<FaceSample> m_samples;

    // ── Trained model ──────────────────────────────────────────────────────
    bool   m_trained;
    Vector m_meanFace;           ///< Mean face (D)
    Vector m_eigenvalues;        ///< All eigenvalues sorted descending
    Matrix m_eigenfaces;         ///< Top K eigenfaces, each row is an eigenface (K × D)
    size_t m_K;                  ///< Number of eigenfaces selected

    /// Projected training data: (weights, label) for each training image
    std::vector<std::pair<Vector, std::string>> m_trainingProjections;
};

} // namespace facerecog

#endif // PCA_PIPELINE_H
