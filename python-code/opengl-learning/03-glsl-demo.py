#coding:utf-8

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
#from numpy import *
import sys

def InitGLUT():
    glutInit(sys.argv)
    glutWindow = glutCreateWindow(b"GPGPU")

def InitFBO(nWidth,intnHeight,pFb):
    #创建FBO并绑定

    glewInit()
    fb = glGenFramebuffersEXT(1)
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,fb)
    pFb = fb
    #用绘制来调用
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0,nWidth,0,nHeight)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glViewport(0,0,nWidth,nHeight)
    return fb