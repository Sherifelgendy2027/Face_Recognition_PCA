/**
 * @file main.cpp
 * @brief Test harness for MathEngine + PCA Pipeline — Steps 1 & 2 validation.
 *
 * Runs self-tests for:
 *   - MathEngine: matrix ops, vectors, covariance, eigendecomposition
 *   - PCA_Pipeline: dataset loading, training, projection, reconstruction
 *
 * Usage:
 *   ./FaceRecognition                    → runs MathEngine tests only
 *   ./FaceRecognition /path/to/dataset   → runs MathEngine + PCA Pipeline tests
 */

#include "core/MathEngine.h"
#include "pipeline/PCA_Pipeline.h"

#include <QImage>

#include <iostream>
#include <cmath>
#include <string>

using namespace facerecog;

static int g_passed = 0;
static int g_failed = 0;

static void check(bool condition, const std::string& testName)
{
    if (condition) {
        std::cout << "  ✅ PASS: " << testName << std::endl;
        ++g_passed;
    } else {
        std::cout << "  ❌ FAIL: " << testName << std::endl;
        ++g_failed;
    }
}

static bool approxEqual(double a, double b, double tol = 1e-6)
{
    return std::fabs(a - b) < tol;
}

// ═══════════════════════════════════════════════════════════════════════
// MathEngine Tests
// ═══════════════════════════════════════════════════════════════════════

static void testConstruction()
{
    std::cout << "\n── Matrix Construction ──" << std::endl;
    auto Z = MathEngine::zeros(3, 4);
    check(Z.size() == 3 && Z[0].size() == 4 && Z[2][3] == 0.0, "zeros(3,4)");
    auto I = MathEngine::identity(3);
    check(I[0][0] == 1.0 && I[1][1] == 1.0 && I[2][2] == 1.0
          && I[0][1] == 0.0, "identity(3)");
}

static void testTranspose()
{
    std::cout << "\n── Transpose ──" << std::endl;
    Matrix A = {{1, 2, 3}, {4, 5, 6}};
    auto T = MathEngine::transpose(A);
    check(T.size() == 3 && T[0].size() == 2, "transpose dimensions");
    check(T[0][0] == 1 && T[0][1] == 4 && T[2][0] == 3 && T[2][1] == 6,
          "transpose values");
}

static void testMultiply()
{
    std::cout << "\n── Matrix Multiplication ──" << std::endl;
    Matrix A = {{1, 2}, {3, 4}};
    Matrix B = {{5, 6}, {7, 8}};
    auto C = MathEngine::multiply(A, B);
    check(C.size() == 2 && C[0].size() == 2, "multiply dimensions");
    check(approxEqual(C[0][0], 19) && approxEqual(C[0][1], 22)
          && approxEqual(C[1][0], 43) && approxEqual(C[1][1], 50),
          "multiply values (2x2 * 2x2)");

    Matrix D = {{1, 2, 3}, {4, 5, 6}};
    Matrix E = {{7}, {8}, {9}};
    auto F = MathEngine::multiply(D, E);
    check(F.size() == 2 && F[0].size() == 1, "multiply non-square dims");
    check(approxEqual(F[0][0], 50) && approxEqual(F[1][0], 122),
          "multiply non-square values");

    auto I = MathEngine::identity(2);
    auto AI = MathEngine::multiply(A, I);
    check(approxEqual(AI[0][0], 1) && approxEqual(AI[1][1], 4), "A * I = A");
}

static void testArithmetic()
{
    std::cout << "\n── Element-wise Arithmetic ──" << std::endl;
    Matrix A = {{1, 2}, {3, 4}};
    Matrix B = {{5, 6}, {7, 8}};
    auto sum = MathEngine::add(A, B);
    check(approxEqual(sum[0][0], 6) && approxEqual(sum[1][1], 12), "add");
    auto diff = MathEngine::subtract(B, A);
    check(approxEqual(diff[0][0], 4) && approxEqual(diff[1][1], 4), "subtract");
    auto scaled = MathEngine::multiplyScalar(A, 2.0);
    check(approxEqual(scaled[0][0], 2) && approxEqual(scaled[1][1], 8),
          "scalar multiply");
}

static void testVectorOps()
{
    std::cout << "\n── Vector Operations ──" << std::endl;
    Vector a = {1, 2, 3};
    Vector b = {4, 5, 6};
    check(approxEqual(MathEngine::dot(a, b), 32.0), "dot product");
    check(approxEqual(MathEngine::norm(a), std::sqrt(14.0)), "norm");
    check(approxEqual(MathEngine::euclideanDistance(a, b),
                      std::sqrt(27.0)), "euclidean distance");
    auto n = MathEngine::normalize(a);
    check(approxEqual(MathEngine::norm(n), 1.0), "normalize → unit length");
    Vector z = {0, 0, 0};
    auto nz = MathEngine::normalize(z);
    check(approxEqual(MathEngine::norm(nz), 0.0), "normalize zero vector");
}

static void testStatistics()
{
    std::cout << "\n── Statistical Operations ──" << std::endl;
    std::vector<Vector> vecs = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto mean = MathEngine::meanVector(vecs);
    check(approxEqual(mean[0], 4) && approxEqual(mean[1], 5)
          && approxEqual(mean[2], 6), "mean vector");

    Matrix data = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto centered = MathEngine::subtractVectorFromRows(data, mean);
    check(approxEqual(centered[0][0], -3) && approxEqual(centered[1][0], 0)
          && approxEqual(centered[2][0], 3), "subtract mean from rows");

    auto cov = MathEngine::covarianceMatrix(centered);
    check(cov.size() == 3 && cov[0].size() == 3, "covariance dimensions");
    check(approxEqual(cov[0][0], 9.0) && approxEqual(cov[1][1], 9.0),
          "covariance diagonal values");
    check(approxEqual(cov[0][1], 9.0),
          "covariance off-diagonal (perfect correlation)");

    auto smallCov = MathEngine::smallCovarianceMatrix(centered);
    check(smallCov.size() == 3 && smallCov[0].size() == 3,
          "small covariance dimensions (NxN)");
}

static void testEigen()
{
    std::cout << "\n── Eigendecomposition ──" << std::endl;
    Matrix S = {{2, 1}, {1, 2}};
    auto result = MathEngine::eigenDecomposition(S);
    check(result.eigenvalues.size() == 2, "eigen: 2 eigenvalues");
    check(approxEqual(result.eigenvalues[0], 3.0, 1e-4), "eigen: λ₁ ≈ 3");
    check(approxEqual(result.eigenvalues[1], 1.0, 1e-4), "eigen: λ₂ ≈ 1");

    double dotProd = MathEngine::dot(result.eigenvectors[0], result.eigenvectors[1]);
    check(approxEqual(dotProd, 0.0, 1e-4), "eigen: eigenvectors orthogonal");
    check(approxEqual(MathEngine::norm(result.eigenvectors[0]), 1.0, 1e-4),
          "eigen: eigenvector 1 unit length");

    Matrix v0 = {{result.eigenvectors[0][0]}, {result.eigenvectors[0][1]}};
    auto Av = MathEngine::multiply(S, v0);
    check(approxEqual(Av[0][0], 3.0 * v0[0][0], 1e-4)
          && approxEqual(Av[1][0], 3.0 * v0[1][0], 1e-4),
          "eigen: A·v₁ = λ₁·v₁");

    Matrix D = {{5, 0, 0}, {0, 3, 0}, {0, 0, 1}};
    auto rd = MathEngine::eigenDecomposition(D);
    check(approxEqual(rd.eigenvalues[0], 5.0, 1e-4)
          && approxEqual(rd.eigenvalues[1], 3.0, 1e-4)
          && approxEqual(rd.eigenvalues[2], 1.0, 1e-4),
          "eigen: diagonal matrix eigenvalues");
}

static void testSelectK()
{
    std::cout << "\n── K Selection ──" << std::endl;
    Vector eigenvals = {50, 30, 10, 5, 3, 2};
    check(MathEngine::selectK(eigenvals, 0.95) == 4, "selectK(95%) → 4");
    check(MathEngine::selectK(eigenvals, 0.80) == 2, "selectK(80%) → 2");
    check(MathEngine::selectK(eigenvals, 1.0) == 6, "selectK(100%) → 6");
}

static void testUtility()
{
    std::cout << "\n── Utility ──" << std::endl;
    Matrix M = {{1, 2, 3}, {4, 5, 6}};
    auto flat = MathEngine::flatten(M);
    check(flat.size() == 6 && approxEqual(flat[0], 1) && approxEqual(flat[5], 6),
          "flatten 2x3 → 6 elements");
    auto reshaped = MathEngine::reshape(flat, 2, 3);
    check(reshaped.size() == 2 && reshaped[0].size() == 3
          && approxEqual(reshaped[1][2], 6), "reshape 6 → 2x3");
}

// ═══════════════════════════════════════════════════════════════════════
// PCA Pipeline Tests (require dataset path as CLI argument)
// ═══════════════════════════════════════════════════════════════════════

static void testImageConversion()
{
    std::cout << "\n── Image ↔ Vector Conversion ──" << std::endl;

    // Create a synthetic grayscale image
    QImage synth(PCA_Pipeline::FACE_WIDTH, PCA_Pipeline::FACE_HEIGHT,
                 QImage::Format_Grayscale8);
    synth.fill(128); // uniform gray

    auto vec = PCA_Pipeline::imageToVector(synth);
    check(static_cast<int>(vec.size()) == PCA_Pipeline::FACE_DIM,
          "imageToVector: correct dimension");
    check(approxEqual(vec[0], 128.0) && approxEqual(vec[vec.size() - 1], 128.0),
          "imageToVector: pixel values preserved");

    // Round-trip: vector → image → vector
    auto img2 = PCA_Pipeline::vectorToImage(vec);
    check(img2.width() == PCA_Pipeline::FACE_WIDTH
          && img2.height() == PCA_Pipeline::FACE_HEIGHT,
          "vectorToImage: correct size");
    auto vec2 = PCA_Pipeline::imageToVector(img2);
    check(approxEqual(vec2[0], 128.0), "round-trip: pixel value preserved");
}

static void testPCAPipeline(const std::string& datasetPath)
{
    std::cout << "\n── PCA Pipeline: Dataset Loading ──" << std::endl;

    PCA_Pipeline pca;
    size_t count = pca.loadDataset(datasetPath);
    check(count > 0, "loaded > 0 images");
    check(pca.getNumClasses() > 0, "detected > 0 identities");

    std::cout << "  → " << count << " images, "
              << pca.getNumClasses() << " classes" << std::endl;

    // Verify sample dimensions
    const auto& samples = pca.getSamples();
    check(static_cast<int>(samples[0].pixels.size()) == PCA_Pipeline::FACE_DIM,
          "sample dimension = " + std::to_string(PCA_Pipeline::FACE_DIM));
    check(!samples[0].label.empty(), "sample has non-empty label");

    // ── Training ──
    std::cout << "\n── PCA Pipeline: Training ──" << std::endl;
    auto result = pca.train(0.95);

    check(pca.isTrained(), "model is trained");
    check(result.selectedK > 0, "K > 0 eigenfaces selected");
    check(result.varianceCaptured >= 0.95,
          "variance captured >= 95% (actual: " +
          std::to_string(result.varianceCaptured * 100) + "%)");
    check(result.numSamples == count, "result.numSamples matches loaded count");
    check(result.dimension == static_cast<size_t>(PCA_Pipeline::FACE_DIM),
          "result.dimension = FACE_DIM");

    std::cout << "  → K=" << result.selectedK
              << ", variance=" << (result.varianceCaptured * 100) << "%"
              << std::endl;

    // Mean face sanity: all values should be in [0, 255]
    const auto& meanFace = pca.getMeanFace();
    bool meanInRange = true;
    for (double v : meanFace) {
        if (v < 0.0 || v > 255.0) { meanInRange = false; break; }
    }
    check(meanInRange, "mean face values in [0, 255]");

    // Eigenfaces should be unit vectors
    const auto& eigenfaces = pca.getEigenfaces();
    check(eigenfaces.size() == result.selectedK, "eigenfaces count = K");
    double ef_norm = MathEngine::norm(eigenfaces[0]);
    check(approxEqual(ef_norm, 1.0, 1e-3), "first eigenface is unit vector");

    // ── Projection & Reconstruction ──
    std::cout << "\n── PCA Pipeline: Projection & Reconstruction ──" << std::endl;

    // Project a training face → reconstruct → measure error
    const Vector& firstFace = samples[0].pixels;
    Vector weights = pca.project(firstFace);
    check(weights.size() == result.selectedK,
          "projection dimensionality = K");

    Vector reconstructed = pca.reconstruct(weights);
    check(static_cast<int>(reconstructed.size()) == PCA_Pipeline::FACE_DIM,
          "reconstructed dimensionality = FACE_DIM");

    // Reconstruction error for a training face should be relatively small
    double error = pca.reconstructionError(firstFace);
    std::cout << "  → Reconstruction error (training face): " << error << std::endl;
    check(error < 1e7, "reconstruction error is bounded for training face");

    // Error for a non-face (all white) should be much larger
    Vector whiteFace(PCA_Pipeline::FACE_DIM, 255.0);
    double whiteError = pca.reconstructionError(whiteFace);
    std::cout << "  → Reconstruction error (white image):   " << whiteError << std::endl;
    check(whiteError > error,
          "non-face has higher reconstruction error than training face");

    // Training projections should be available
    const auto& projections = pca.getTrainingProjections();
    check(projections.size() == count,
          "training projections count = sample count");
    check(projections[0].first.size() == result.selectedK,
          "projection weight dimension = K");
    check(!projections[0].second.empty(),
          "projection label is non-empty");

    // Save mean face as image for visual inspection
    QImage meanImg = PCA_Pipeline::vectorToImage(meanFace);
    QString meanPath = QString::fromStdString(datasetPath) + "/../mean_face.png";
    if (meanImg.save(meanPath)) {
        std::cout << "  → Mean face saved to: "
                  << meanPath.toStdString() << std::endl;
    }
}

// ═══════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[])
{
    std::cout << "╔══════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Face Recognition Test Suite — Steps 1 & 2  ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════╝" << std::endl;

    // ── MathEngine Tests ──
    std::cout << "\n━━━ MathEngine Tests ━━━" << std::endl;
    testConstruction();
    testTranspose();
    testMultiply();
    testArithmetic();
    testVectorOps();
    testStatistics();
    testEigen();
    testSelectK();
    testUtility();

    // ── Image Conversion Tests (no dataset needed) ──
    std::cout << "\n━━━ Image Conversion Tests ━━━" << std::endl;
    testImageConversion();

    // ── PCA Pipeline Tests (need dataset path) ──
    if (argc >= 2) {
        std::string datasetPath = argv[1];
        std::cout << "\n━━━ PCA Pipeline Tests ━━━" << std::endl;
        std::cout << "  Dataset: " << datasetPath << std::endl;
        try {
            testPCAPipeline(datasetPath);
        } catch (const std::exception& e) {
            std::cout << "  ❌ EXCEPTION: " << e.what() << std::endl;
            ++g_failed;
        }
    } else {
        std::cout << "\n━━━ PCA Pipeline Tests: SKIPPED ━━━" << std::endl;
        std::cout << "  (Pass dataset path as argument to run PCA tests)" << std::endl;
        std::cout << "  Usage: ./FaceRecognition /path/to/dataset" << std::endl;
    }

    // ── Summary ──
    std::cout << "\n══════════════════════════════════════════════" << std::endl;
    std::cout << "  Results: " << g_passed << " passed, "
              << g_failed << " failed" << std::endl;
    std::cout << "══════════════════════════════════════════════\n" << std::endl;

    return g_failed > 0 ? 1 : 0;
}
