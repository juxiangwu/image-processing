TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

win32: LIBS += -LC:/Intel/computer_vision_sdk_2018.3.343/openvx/lib/ \
-lopenvx

win32: LIBS += -LC:/Intel/computer_vision_sdk_2018.3.343/opencv/x64/vc14/lib \
-lopencv_pvl343 \
-lopencv_world343

INCLUDEPATH += C:/Intel/computer_vision_sdk_2018.3.343/openvx/include
DEPENDPATH += C:/Intel/computer_vision_sdk_2018.3.343/openvx/include
INCLUDEPATH += C:/Intel/computer_vision_sdk_2018.3.343/opencv/include
DEPENDPATH += C:/Intel/computer_vision_sdk_2018.3.343/opencv/include

HEADERS += \
    opencv_camera_display.h
