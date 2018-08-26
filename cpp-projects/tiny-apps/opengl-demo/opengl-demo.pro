TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

win32: LIBS += -L$$PWD/../../../../../../../temp/lib/x64/ \
-lfreeglut \
-lglew32

INCLUDEPATH += $$PWD/../../../../../../../temp/inc
DEPENDPATH += $$PWD/../../../../../../../temp/inc
