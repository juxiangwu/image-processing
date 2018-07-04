#coding:utf-8

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from numpy import *
import sys
 
global W, H, R
(W, H, R) = (500, 500, 10.0)
 
def init():
    glClearColor(1.0, 1.0, 1.0, 1.0)
    
def drawfunc():
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(0.0, 0.0, 0.0)
    glBegin(GL_POINTS)
    for x in arange(-R, R, 0.04):
#         print '%.1f%%r' % ((R + x) / (R + R) * 100),
        for y in arange(-R, R, 0.04):
            r = cos(x) + sin(y)
            glColor3f(cos(y*r), cos(x*y*r), sin(x*r))
            glVertex2f(x, y)
#     print '100%!!'
    glEnd()
    glFlush()

def reshape(w, h):
    if h <= 0: h = 1;
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if w <= h:
        gluOrtho2D(-R, R, -R * h / w, R * h / w)
    else:
        gluOrtho2D(-R * w / h, R * w / h, -R, R)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def keyboard(key, x, y):
    # print('key press:',key)
    key = key.decode('utf-8')
    if key == chr(27) or key == "q": # Esc is 27
        sys.exit()

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_SINGLE|GLUT_RGB)
    glutInitWindowPosition(20, 20)
    glutInitWindowSize(W, H)
    glutCreateWindow(b"Artist Drawer")
    glutReshapeFunc(reshape)
    glutDisplayFunc(drawfunc)
    glutKeyboardFunc(keyboard)
    init()
    glutMainLoop()

main()