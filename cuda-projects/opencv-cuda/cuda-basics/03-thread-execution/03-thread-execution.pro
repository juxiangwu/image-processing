TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES +=

HEADERS += \
    main.cu

DESTDIR = ../bin
CUDA_OBJECTS_DIR = OBJECTS_DIR/../cuda
# This makes the .cu files appear in your project
OTHER_FILES += \
    main.cu \

CUDA_SOURCES += \
    main.cu

# MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
MSVCRT_LINK_FLAG_DEBUG   = "/MDd"
MSVCRT_LINK_FLAG_RELEASE = "/MD"

# CUDA settings
CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.1/"                # Path to cuda toolkit install
SYSTEM_NAME = x64                 # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64                    # '32' or '64', depending on your system
CUDA_ARCH = sm_50                   # Type of CUDA architecture
NVCC_OPTIONS = --use_fast_math

# include paths
INCLUDEPATH += $$CUDA_DIR/include \
               $$CUDA_DIR/common/inc \
               $$CUDA_DIR/../shared/inc \
               "C:/Program Files/glm/include"

INCLUDEPATH += $$PWD/../../../../../libs/opencv-3.4.3-cu91-vs2015-x64/include
DEPENDPATH += $$PWD/../../../../../libs/opencv-3.4.3-cu91-vs2015-x64/include
# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME \
                $$CUDA_DIR/common/lib/$$SYSTEM_NAME \
                $$CUDA_DIR/../shared/lib/$$SYSTEM_NAME
# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

# Add the necessary libraries
CUDA_LIB_NAMES = cudart_static kernel32 user32 gdi32 winspool comdlg32 \
                 advapi32 shell32 ole32 oleaut32 uuid odbc32 odbccp32 \
                 #freeglut glew32

for(lib, CUDA_LIB_NAMES) {
    CUDA_LIBS += -l$$lib
}
LIBS += $$CUDA_LIBS

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
    cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
                      --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                      --compile -cudart static -g -DWIN32 -D_MBCS \
                      -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1" \
                      -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG \
                      -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
    cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
                    --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                    --compile -cudart static -DWIN32 -D_MBCS \
                    -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi" \
                    -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
                    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_CPP
    QMAKE_EXTRA_COMPILERS += cuda
}

win32: LIBS += -L$$PWD/../../../../../../libs/opencv-3.4.3-cu91-vs2015-x64/x64/vc14/lib/ \
-lcvBlob \
-lopencv_aruco343 \
-lopencv_bgsegm343 \
-lopencv_bioinspired343 \
-lopencv_calib3d343 \
-lopencv_ccalib343 \
-lopencv_core343 \
-lopencv_cudaarithm343 \
-lopencv_cudabgsegm343 \
-lopencv_cudacodec343 \
-lopencv_cudafeatures2d343 \
-lopencv_cudafilters343 \
-lopencv_cudaimgproc343 \
-lopencv_cudalegacy343 \
-lopencv_cudaobjdetect343 \
-lopencv_cudaoptflow343 \
-lopencv_cudastereo343 \
-lopencv_cudawarping343 \
-lopencv_cudev343 \
-lopencv_datasets343 \
-lopencv_dnn343 \
-lopencv_dnn_objdetect343 \
-lopencv_dpm343 \
-lopencv_face343 \
-lopencv_features2d343 \
-lopencv_flann343 \
-lopencv_freetype343 \
-lopencv_fuzzy343 \
-lopencv_hdf343 \
-lopencv_hfs343 \
-lopencv_highgui343 \
-lopencv_imgcodecs343 \
-lopencv_imgproc343 \
-lopencv_img_hash343 \
-lopencv_line_descriptor343 \
-lopencv_ml343 \
-lopencv_objdetect343 \
-lopencv_optflow343 \
-lopencv_phase_unwrapping343 \
-lopencv_photo343 \
-lopencv_plot343 \
-lopencv_reg343 \
-lopencv_rgbd343 \
-lopencv_saliency343 \
-lopencv_sfm343 \
-lopencv_shape343 \
-lopencv_stereo343 \
-lopencv_stitching343 \
-lopencv_structured_light343 \
-lopencv_superres343 \
-lopencv_surface_matching343 \
-lopencv_text343 \
-lopencv_tracking343 \
-lopencv_video343 \
-lopencv_videoio343 \
-lopencv_videostab343 \
-lopencv_viz343 \
-lopencv_xfeatures2d343 \
-lopencv_ximgproc343 \
-lopencv_xobjdetect343 \
-lopencv_xphoto343 \
-ldlib19.15.99_release_64bit_msvc1900

INCLUDEPATH += $$PWD/../../../../../../libs/opencv-3.4.3-cu91-vs2015-x64/include
DEPENDPATH += $$PWD/../../../../../../libs/opencv-3.4.3-cu91-vs2015-x64/include
