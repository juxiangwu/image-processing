#coding:utf-8

import pygame
import OpenGL
OpenGL.USE_ACCELERATOR = True
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import sys
# from ADParser import parse_file

def print_openGL_info():
    # Query OpenGL for version information:
    print ('glGetIntegerv(GL_MAJOR_VERSION) =',glGetIntegerv(GL_MAJOR_VERSION))
    print ('glGetIntegerv(GL_MINOR_VERSION) =',glGetIntegerv(GL_MINOR_VERSION))
    print ('glGetString(GL_VERSION) = \''+glGetString(GL_VERSION).decode('utf-8')+'\'')
    print ('glGetString(GL_VENDOR) = \''+glGetString(GL_VENDOR).decode('utf-8')+'\'')
    print ('glGetString(GL_RENDERER) = \''+glGetString(GL_RENDERER).decode('utf-8')+'\'')
    n_extensions = glGetIntegerv(GL_NUM_EXTENSIONS)
    print ('glGetIntegerv(GL_NUM_EXTENSIONS) =',n_extensions)
    for i in range(n_extensions):
        print ('glGetStringi(GL_EXTENSIONS, '+str(i)+') = \''+glGetStringi(GL_EXTENSIONS, i).decode('utf-8')+'\'')
    print ('glGetString(GL_SHADING_LANGUAGE_VERSION) = \''+glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8')+'\'')
    # The following doesn't seem to be supported by pyOpenGL:
    # n_supported_GLSL_vers = glGetIntegerv(GL_NUM_SHADING_LANGUAGE_VERSIONS)
    # print 'glGetIntegerv(GL_NUM_SHADING_LANGUAGE_VERSIONS) =',n_supported_GLSL_vers
        # for i in xrange(n_supported_GLSL_vers):
    # print 'glGetStringi(GL_SHADING_LANGUAGE_VERSION, '+str(i)+') = \''+glGetStringi(GL_SHADING_LANGUAGE_VERSION, i)+'\''
    
if __name__=="__main__":
    pygame.init()
    display = (100,100)
    pygame.display.set_mode(display, pygame.DOUBLEBUF|pygame.OPENGL)
    print_openGL_info()