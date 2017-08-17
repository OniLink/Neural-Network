TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt
LIBS += -lsfml-audio -fopenmp

SOURCES += \
	jsoncpp.cpp \
    Audio.cpp

HEADERS += \
    NetworkLayer.hpp \
    Vector.hpp \
    Matrix.hpp \
    FeedForwardLayer.hpp \
    NeuralNetwork.hpp \
    LSTMLayer.hpp \
    FFT.hpp \
    json/json-forwards.h \
	json/json.h
