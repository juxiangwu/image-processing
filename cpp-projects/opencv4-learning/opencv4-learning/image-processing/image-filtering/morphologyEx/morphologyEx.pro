TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp
win32: LIBS += -L$$PWD/../../../../../../../../temp/opencv/build/install/x64/vc14/lib/ \
 -lopencv_aruco400 \
 -lopencv_bgsegm400 \
 -lopencv_bioinspired400 \
 -lopencv_calib3d400 \
 -lopencv_ccalib400 \
 -lopencv_core400 \
 -lopencv_cudaarithm400 \
 -lopencv_cudabgsegm400 \
 -lopencv_cudacodec400 \
 -lopencv_cudafeatures2d400 \
 -lopencv_cudafilters400 \
 -lopencv_cudaimgproc400 \
 -lopencv_cudalegacy400 \
 -lopencv_cudaobjdetect400 \
 -lopencv_cudaoptflow400 \
 -lopencv_cudastereo400 \
 -lopencv_cudawarping400 \
 -lopencv_cudev400 \
 -lopencv_datasets400 \
 -lopencv_dnn400 \
 -lopencv_dnn_objdetect400 \
 -lopencv_dpm400 \
 -lopencv_face400 \
 -lopencv_features2d400 \
 -lopencv_flann400 \
 -lopencv_freetype400 \
 -lopencv_fuzzy400 \
 -lopencv_hdf400 \
 -lopencv_hfs400 \
 -lopencv_highgui400 \
 -lopencv_imgcodecs400 \
 -lopencv_imgproc400 \
 -lopencv_img_hash400 \
 -lopencv_line_descriptor400 \
 -lopencv_ml400 \
 -lopencv_objdetect400 \
 -lopencv_optflow400 \
 -lopencv_phase_unwrapping400 \
 -lopencv_photo400 \
 -lopencv_plot400 \
 -lopencv_reg400 \
 -lopencv_rgbd400 \
 -lopencv_saliency400 \
 -lopencv_sfm400 \
 -lopencv_shape400 \
 -lopencv_stereo400 \
 -lopencv_stitching400 \
 -lopencv_structured_light400 \
 -lopencv_superres400 \
 -lopencv_surface_matching400 \
 -lopencv_text400 \
 -lopencv_tracking400 \
 -lopencv_video400 \
 -lopencv_videoio400 \
 -lopencv_videostab400 \
 -lopencv_viz400 \
 -lopencv_xfeatures2d400 \
 -lopencv_ximgproc400 \
 -lopencv_xobjdetect400 \
 -lopencv_xphoto400

INCLUDEPATH += $$PWD/../../../../../../../../temp/opencv/build/install/include
DEPENDPATH += $$PWD/../../../../../../../../temp/opencv/build/install/include
