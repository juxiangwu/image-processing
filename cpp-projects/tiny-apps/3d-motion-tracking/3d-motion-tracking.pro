TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp \
    src/Alignment.cpp \
    src/Distribution.cpp \
    src/FilterTransformation.cpp \
    src/Grid.cpp \
    src/Hungarian.cpp \
    src/IdentityTransformation.cpp \
    src/main.cpp \
    src/MeshObject.cpp \
    src/Particle.cpp \
    src/ParticleTracker.cpp \
    src/TrackUtils.cpp \
    src/VideoCamera.cpp \
    common.cpp \
    main.cpp

win32: LIBS += -L$$PWD/../../../../../libs/opencv-3.4.2-vs2015x64-qt/x64/vc14/lib/ \
-lopencv_aruco342 \
-lopencv_bgsegm342 \
-lopencv_bioinspired342 \
-lopencv_calib3d342 \
-lopencv_ccalib342 \
-lopencv_core342 \
-lopencv_cudaarithm342 \
-lopencv_cudabgsegm342 \
-lopencv_cudacodec342 \
-lopencv_cudafeatures2d342 \
-lopencv_cudafilters342 \
-lopencv_cudaimgproc342 \
-lopencv_cudalegacy342 \
-lopencv_cudaobjdetect342 \
-lopencv_cudaoptflow342 \
-lopencv_cudastereo342 \
-lopencv_cudawarping342 \
-lopencv_cudev342 \
-lopencv_datasets342 \
-lopencv_dnn342 \
-lopencv_dnn_objdetect342 \
-lopencv_dpm342 \
-lopencv_face342 \
-lopencv_features2d342 \
-lopencv_flann342 \
-lopencv_freetype342 \
-lopencv_fuzzy342 \
-lopencv_hdf342 \
-lopencv_hfs342 \
-lopencv_highgui342 \
-lopencv_imgcodecs342 \
-lopencv_imgproc342 \
-lopencv_img_hash342 \
-lopencv_line_descriptor342 \
-lopencv_ml342 \
-lopencv_objdetect342 \
-lopencv_optflow342 \
-lopencv_phase_unwrapping342 \
-lopencv_photo342 \
-lopencv_plot342 \
-lopencv_reg342 \
-lopencv_rgbd342 \
-lopencv_saliency342 \
-lopencv_sfm342 \
-lopencv_shape342 \
-lopencv_stereo342 \
-lopencv_stitching342 \
-lopencv_structured_light342 \
-lopencv_superres342 \
-lopencv_surface_matching342 \
-lopencv_text342 \
-lopencv_tracking342 \
-lopencv_video342 \
-lopencv_videoio342 \
-lopencv_videostab342 \
-lopencv_viz342 \
-lopencv_xfeatures2d342 \
-lopencv_ximgproc342 \
-lopencv_xobjdetect342 \
-lopencv_xphoto342 \
-ldlib_release_64bit_msvc1900 \
-lfreeglut \
-lglew32 \
-lopengl32 \
-lglfw3dll \
-lglog

INCLUDEPATH += $$PWD/../../../../../libs/opencv-3.4.2-vs2015x64-qt/include \
               C:/local/boost_1_61_0 \
               $$PWD/include

DEPENDPATH += $$PWD/../../../../../libs/opencv-3.4.2-vs2015x64-qt/include \
               C:/local/boost_1_61_0 \
               $$PWD/include

HEADERS += \
    similiar_process.h \
    include/Alignment.h \
    include/CalcUtils.h \
    include/coloriser.h \
    include/Distribution.h \
    include/DrawUtils.h \
    include/FilterTransformation.h \
    include/Grid.h \
    include/Hungarian.h \
    include/IdentityTransformation.h \
    include/ITracker.h \
    include/ITransformation.h \
    include/IVideoSource.h \
    include/master.h \
    include/MeshObject.h \
    include/Particle.h \
    include/ParticleTracker.h \
    include/Trackable.h \
    include/TrackUtils.h \
    include/VideoCamera.h \
    common.h \
    TestHungarian.h

