# CV Task 5 — Native C++ PCA Face Detection & Recognition

A native C++ / Qt Widgets desktop application for face detection and recognition using Eigenfaces/PCA from scratch.

## Constraints respected

- No OpenCV
- No Dlib
- No dedicated computer-vision libraries
- Qt is used only for desktop UI and image loading/cropping/scaling
- Eigen is used only as the math backend for eigendecomposition inside `MathEngine`

## Implemented steps

### Step 1 — Core Math Engine

Located in `src/core/`.

Provides:

- Matrix and vector operations
- Mean vector calculation
- Covariance and small covariance matrix calculation
- Eigendecomposition through Eigen
- K-selection by cumulative variance

### Step 2 — PCA Pipeline

Located in `src/pipeline/PCAPipeline.*`.

Provides:

- Dataset loading from numerically named AT&T/ORL folders
- Dynamic PCA training window detection from the first valid training image
- Grayscale image flattening to `trainWidth × trainHeight` vectors
- Mean-face calculation
- Turk-Pentland PCA optimization using `A * A^T`
- Dynamic eigenface selection for the requested variance, default `95%`
- Projection and reconstruction APIs

During dataset loading, the application logs the detected PCA window, for example:

```text
[PCAPipeline] Dynamic PCA training window detected from first image: 92x112 (D=10304)
```

### Step 3 — Face Detection

Located in `src/pipeline/FaceDetector.*`.

Provides:

- Native sliding-window scan over a target image
- PCA projection per candidate window
- Reconstruction RMSE thresholding
- Non-maximum suppression
- Bounding box output in original raw-image coordinates

If the target image is smaller than the dynamic PCA training window, the detector now scales the image upward with `QImage::scaled(..., Qt::KeepAspectRatio, Qt::SmoothTransformation)` instead of black-padding it. This keeps the candidate patch visually meaningful for PCA projection.

### Step 4 — Face Recognition

Located in `src/pipeline/FaceRecognizer.*`.

Provides:

- Native K-nearest-neighbors classifier
- Euclidean distance in PCA face space
- Identity label, nearest distance, and confidence output

### Step 5 — Qt UI + ROC Plotting

Located in `src/ui/` and `src/main.cpp`.

Provides:

- Qt Widgets desktop UI
- Dataset training button
- Test image loading button
- Detection and recognition visualization
- Native `QPainter` ROC widget without QtCharts

## Project structure

```text
CV_Task_5/
├── build/
├── dataset/
│   ├── 1/
│   ├── 2/
│   └── ...
├── src/
│   ├── core/
│   │   ├── MathEngine.h
│   │   └── MathEngine.cpp
│   ├── pipeline/
│   │   ├── PCAPipeline.h
│   │   ├── PCAPipeline.cpp
│   │   ├── FaceDetector.h
│   │   ├── FaceDetector.cpp
│   │   ├── FaceRecognizer.h
│   │   └── FaceRecognizer.cpp
│   ├── ui/
│   │   ├── MainWindow.h
│   │   ├── MainWindow.cpp
│   │   ├── ROCWidget.h
│   │   └── ROCWidget.cpp
│   └── main.cpp
├── CMakeLists.txt
└── README.md
```

## Build

From the project root:

```bash
cmake -S . -B build
cmake --build build
```

Run from the build directory so the default dataset path resolves to `../dataset`:

```bash
cd build
./CV_Task_5
```

On Windows, run the generated executable from the `build` output folder and select the dataset manually if needed.

## Usage

1. Click **Select Dataset & Train PCA**.
2. Select the `dataset/` folder in the project root.
3. Check the log for the dynamic PCA window size.
4. Load a test image.
5. Click **Detect & Recognize**.

For single AT&T/ORL face images, keep `Max scale = 1.0`. For larger images, increase `Max scale` and adjust the stride.

## Architecture Update: Dynamic PCA Window + Proper Resizing

The previous small-image safety behavior used black padding. That allowed the loops to run, but it damaged the PCA projection because the black region became part of the face vector.

The updated behavior is:

- `PCAPipeline` detects `trainWidth` and `trainHeight` from the first valid training image.
- All training images are converted to the same dynamic geometry.
- Projection, reconstruction, detection, recognition, and RMSE normalization use `trainWidth × trainHeight` instead of hardcoded `92 × 112`.
- The UI keeps the loaded target image as a raw, unscaled `QImage` for detection and recognition.
- QLabel scaling is display-only.
- `FaceDetector` scales a too-small target image upward with Qt smooth resizing, maps detection boxes back to original raw-image coordinates, and never black-pads.
