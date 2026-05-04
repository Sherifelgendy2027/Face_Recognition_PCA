cmake_minimum_required(VERSION 3.16)
project(CV_Task_5 VERSION 1.0 LANGUAGES CXX)

# ── C++ Standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Qt Auto-tools ───────────────────────────────────────────────────────────
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ── Dependencies ────────────────────────────────────────────────────────────
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Gui Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Widgets)
find_package(Eigen3 REQUIRED)

# ── Source files ────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/core/MathEngine.cpp
    src/pipeline/PCAPipeline.cpp
    src/pipeline/FaceDetector.cpp
    src/pipeline/FaceRecognizer.cpp
    src/ui/MainWindow.cpp
    src/ui/ROCWidget.cpp
)

set(HEADERS
    src/core/MathEngine.h
    src/pipeline/PCAPipeline.h
    src/pipeline/FaceDetector.h
    src/pipeline/FaceRecognizer.h
    src/ui/MainWindow.h
    src/ui/ROCWidget.h
)

# ── Build executable ────────────────────────────────────────────────────────
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Widgets
    Eigen3::Eigen
)

if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ── macOS bundle settings ──────────────────────────────────────────────────
set_target_properties(${PROJECT_NAME} PROPERTIES
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_GUI_IDENTIFIER com.cvtask5.facerecognition
    MACOSX_BUNDLE_BUNDLE_NAME "CV Task 5 Face Recognition"
)
