#ifndef MATHENGINE_H
#define MATHENGINE_H

/**
 * @file MathEngine.h
 * @brief Core mathematical operations engine for the PCA-based face recognition system.
 *
 * Provides matrix operations, vector arithmetic, statistical functions,
 * and eigendecomposition. Uses std::vector<double> as the primary data type
 * and delegates eigendecomposition to Eigen for performance on large matrices.
 *
 * All methods are static — this is a pure utility class with no state.
 */

#include <vector>
#include <cstddef>
#include <string>
#include <utility> // std::pair

namespace facerecog {

/// A 2D matrix stored in row-major order: matrix[row][col]
using Matrix = std::vector<std::vector<double>>;

/// A 1D vector of doubles
using Vector = std::vector<double>;

class MathEngine {
public:
    // ── Matrix Construction ────────────────────────────────────────────────

    /**
     * @brief Create a zero-filled matrix of given dimensions.
     * @param rows Number of rows
     * @param cols Number of columns
     * @return rows x cols matrix of zeros
     */
    static Matrix zeros(size_t rows, size_t cols);

    /**
     * @brief Create an identity matrix of given size.
     * @param n Dimension (n x n)
     * @return n x n identity matrix
     */
    static Matrix identity(size_t n);

    // ── Matrix Arithmetic ──────────────────────────────────────────────────

    /**
     * @brief Transpose a matrix.
     * @param A Input matrix (m x n)
     * @return Transposed matrix (n x m)
     */
    static Matrix transpose(const Matrix& A);

    /**
     * @brief Multiply two matrices: C = A * B.
     * @param A Left matrix (m x k)
     * @param B Right matrix (k x n)
     * @return Result matrix (m x n)
     * @throws std::invalid_argument if inner dimensions mismatch
     */
    static Matrix multiply(const Matrix& A, const Matrix& B);

    /**
     * @brief Multiply a matrix by a scalar: C = s * A.
     * @param A Input matrix
     * @param scalar Scalar value
     * @return Scaled matrix
     */
    static Matrix multiplyScalar(const Matrix& A, double scalar);

    /**
     * @brief Element-wise addition: C = A + B.
     * @param A First matrix
     * @param B Second matrix
     * @return Sum matrix
     * @throws std::invalid_argument if dimensions mismatch
     */
    static Matrix add(const Matrix& A, const Matrix& B);

    /**
     * @brief Element-wise subtraction: C = A - B.
     * @param A First matrix
     * @param B Second matrix
     * @return Difference matrix
     * @throws std::invalid_argument if dimensions mismatch
     */
    static Matrix subtract(const Matrix& A, const Matrix& B);

    // ── Vector Operations ──────────────────────────────────────────────────

    /**
     * @brief Dot product of two vectors.
     * @param a First vector
     * @param b Second vector
     * @return Scalar dot product
     * @throws std::invalid_argument if sizes mismatch
     */
    static double dot(const Vector& a, const Vector& b);

    /**
     * @brief Euclidean (L2) norm of a vector.
     * @param a Input vector
     * @return ||a||_2
     */
    static double norm(const Vector& a);

    /**
     * @brief Euclidean distance between two vectors.
     * @param a First vector
     * @param b Second vector
     * @return ||a - b||_2
     * @throws std::invalid_argument if sizes mismatch
     */
    static double euclideanDistance(const Vector& a, const Vector& b);

    /**
     * @brief Normalize a vector to unit length.
     * @param a Input vector
     * @return Unit vector in same direction (or zero vector if input is zero)
     */
    static Vector normalize(const Vector& a);

    // ── Statistical Operations ─────────────────────────────────────────────

    /**
     * @brief Compute the mean vector from a collection of vectors.
     *
     * Given N vectors of dimension D, computes the element-wise mean.
     *
     * @param vectors Collection of N vectors, each of dimension D
     * @return Mean vector of dimension D
     * @throws std::invalid_argument if vectors is empty or dimensions are inconsistent
     */
    static Vector meanVector(const std::vector<Vector>& vectors);

    /**
     * @brief Subtract a vector from each row of a matrix.
     *
     * Used to center data: each row gets row[i] = row[i] - mean.
     *
     * @param data Matrix where each row is a data point (N x D)
     * @param mean Vector of dimension D to subtract
     * @return Centered matrix (N x D)
     * @throws std::invalid_argument if dimensions mismatch
     */
    static Matrix subtractVectorFromRows(const Matrix& data, const Vector& mean);

    /**
     * @brief Compute the covariance matrix from centered data.
     *
     * For a centered data matrix A (N x D), computes:
     *   C = (1 / (N-1)) * A^T * A     [D x D covariance]
     *
     * @param centeredData Centered data matrix (N x D), where mean is already subtracted
     * @return Covariance matrix (D x D)
     */
    static Matrix covarianceMatrix(const Matrix& centeredData);

    /**
     * @brief Compute the small covariance matrix using the Turk-Pentland trick.
     *
     * For N images of dimension D where D >> N, instead of computing the
     * D x D matrix A^T * A, we compute the smaller N x N matrix A * A^T.
     * The eigenvectors of the large matrix can then be recovered via:
     *   u_i = A^T * v_i / ||A^T * v_i||
     *
     * @param centeredData Centered data matrix (N x D)
     * @return Small covariance matrix (N x N)
     */
    static Matrix smallCovarianceMatrix(const Matrix& centeredData);

    // ── Eigendecomposition (via Eigen library) ─────────────────────────────

    /**
     * @brief Result of an eigendecomposition.
     *
     * eigenvalues[i] corresponds to eigenvectors[i] (a column vector).
     * Sorted in descending order of eigenvalue magnitude.
     */
    struct EigenResult {
        Vector eigenvalues;      ///< Eigenvalues sorted descending
        Matrix eigenvectors;     ///< Each row is an eigenvector (K x D)
    };

    /**
     * @brief Compute eigendecomposition of a symmetric matrix using Eigen.
     *
     * Returns eigenvalues and eigenvectors sorted in descending order
     * of eigenvalue magnitude.
     *
     * @param symmetricMatrix Input symmetric matrix (n x n)
     * @return EigenResult with sorted eigenvalues and eigenvectors
     * @throws std::invalid_argument if matrix is not square
     */
    static EigenResult eigenDecomposition(const Matrix& symmetricMatrix);

    /**
     * @brief Select top K eigenvectors that capture a target cumulative variance.
     *
     * Iterates through eigenvalues (sorted descending) and accumulates
     * their explained variance ratio until the target is reached.
     *
     * @param eigenvalues Sorted eigenvalues (descending)
     * @param targetVariance Fraction of total variance to capture (e.g., 0.95)
     * @return Number of components K needed
     */
    static size_t selectK(const Vector& eigenvalues, double targetVariance = 0.95);

    // ── Utility ────────────────────────────────────────────────────────────

    /**
     * @brief Flatten a 2D matrix into a 1D vector (row-major).
     * @param mat Input matrix
     * @return 1D vector of all elements
     */
    static Vector flatten(const Matrix& mat);

    /**
     * @brief Reshape a 1D vector into a 2D matrix.
     * @param vec Input vector
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Reshaped matrix
     * @throws std::invalid_argument if vec.size() != rows * cols
     */
    static Matrix reshape(const Vector& vec, size_t rows, size_t cols);

    /**
     * @brief Print a matrix to stdout for debugging.
     * @param A Matrix to print
     * @param name Label for the output
     */
    static void printMatrix(const Matrix& A, const std::string& name = "Matrix");

    /**
     * @brief Print a vector to stdout for debugging.
     * @param v Vector to print
     * @param name Label for the output
     */
    static void printVector(const Vector& v, const std::string& name = "Vector");

private:
    MathEngine() = default; // Prevent instantiation
};

} // namespace facerecog

#endif // MATHENGINE_H
