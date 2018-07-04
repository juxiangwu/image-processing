#coding:utf-8
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
#from numpy import *
import sys
 
def init():
    glClearColor(1.0, 1.0, 1.0, 1.0)
    gluOrtho2D(-5.0, 5.0, -5.0, 5.0)
 
def plotfunc():
    glClear(GL_COLOR_BUFFER_BIT)
    glPointSize(3.0)
 
    glColor3f(1.0, 1.0, 0.0)
    glBegin(GL_LINES)
    glVertex2f(-5.0, 0.0)
    glVertex2f(5.0, 0.0)
    glVertex2f(0.0, 5.0)
    glVertex2f(0.0, -5.0)
    glEnd()
 
    glColor3f(0.0, 0.0, 0.0)
    glBegin(GL_LINES)
    #for x in arange(-5.0, 5.0, 0.1):
    for x in (i * 0.1 for i in range(-50, 50)):
        y = x*x
        glVertex2f(x, y)
    glEnd()
 
    glFlush()
 
def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_SINGLE|GLUT_RGB)
    glutInitWindowPosition(50,50)
    glutInitWindowSize(400,400)
    glutCreateWindow(b"Function Plotter")
    glutDisplayFunc(plotfunc)
    init()
    glutMainLoop()
 
main()