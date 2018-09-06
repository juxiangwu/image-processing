TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    main.cpp \
    facedetectorandtracker.cpp \
    faceswapper.cpp

DEFINES += _TIMESPEC_DEFINED OPENCV

win32: LIBS += -L$$PWD/../../../../../libs/opencv-3.4.3-cu91-vs2015-x64/x64/vc14/lib/ \
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
-ldlib_release_64bit_msvc1900

INCLUDEPATH += $$PWD/../../../../../libs/opencv-3.4.3-cu91-vs2015-x64/include
DEPENDPATH += $$PWD/../../../../../libs/opencv-3.4.3-cu91-vs2015-x64/include

win32: LIBS += -L$$PWD/../../../../../libs/darknet-cpp-dll/bin/Release/ \
-llibYOLOv3SE \

INCLUDEPATH += $$PWD/../../../../../libs/darknet-cpp-dll/include
DEPENDPATH += $$PWD/../../../../../libs/darknet-cpp-dll/include

HEADERS += \
    facedetectorandtracker.h \
    faceswapper.h

#win32: LIBS += -L$$PWD/../../../../../libs/opencv-3.4.3-cu91-vs2015-x64/x64/vc14/lib/ -lopencv_aruco343

#INCLUDEPATH += $$PWD/../../../../../libs/opencv-3.4.3-cu91-vs2015-x64/x64/vc14
#DEPENDPATH += $$PWD/../../../../../libs/opencv-3.4.3-cu91-vs2015-x64/x64/vc14

