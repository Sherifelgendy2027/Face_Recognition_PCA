cmake_minimum_required(VERSION 3.16)
project(FaceRecognition VERSION 1.0 LANGUAGES CXX)

# ── C++17 Standard ──────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Qt Auto-tools ───────────────────────────────────────────────────────────
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ── Find Dependencies ──────────────────────────────────────────────────────
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
find_package(Eigen3 REQUIRED)

# ── Source files ────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/core/MathEngine.cpp
    src/pipeline/PCA_Pipeline.cpp
)

set(HEADERS
    src/core/MathEngine.h
    src/pipeline/PCA_Pipeline.h
)

# ── Build executable ────────────────────────────────────────────────────────
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Eigen3::Eigen
)

# ── macOS bundle settings ──────────────────────────────────────────────────
set_target_properties(${PROJECT_NAME} PROPERTIES
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_GUI_IDENTIFIER com.sbme.facerecognition
    MACOSX_BUNDLE_BUNDLE_NAME "Face Recognition"
)
