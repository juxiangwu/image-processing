#coding:utf-8

import sys
from OpenGL.GL import *
from OpenGL.GLUT import *

def main():
    glutInit(sys.argv)

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(720, 480)

    glutCreateWindow(b'Hello Triangle')

    glutDisplayFunc(onDisplay)
    glutKeyboardFunc(onKeyEvent)

    printSystemGLInfo()

    glutMainLoop()


def printSystemGLInfo():
    print('Vendor: %s' % (glGetString(GL_VENDOR)).decode("utf-8") )
    print('Opengl version: %s' % (glGetString(GL_VERSION).decode("utf-8") ))
    print('GLSL Version: %s' % (glGetString(GL_SHADING_LANGUAGE_VERSION)).decode("utf-8") )
    print('Renderer: %s' % (glGetString(GL_RENDERER)).decode("utf-8") )

def onDisplay():
    # Clear the color and depth buffers
    glClearColor (0.1, 0.2, 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glLoadIdentity()

    # green
    glColor3f(0.0, 0.8, 0.0)

    glBegin(GL_TRIANGLES)
    glVertex2f(0, 0)
    glVertex2f(0.5, 0)
    glVertex2f(0, 0.5)
    glEnd()

    # Copy the off-screen buffer to the screen.
    glutSwapBuffers()

# https://butterflyofdream.wordpress.com/2016/04/27/pyopengl-keyboard-wont-respond/
def onKeyEvent(bkey, x, y):
    # Convert bytes object to string 
    key = bkey.decode("utf-8")
    # Allow to quit by pressing 'Esc'
    if key == chr(27):
        print("Exiting")
        sys.exit()

if __name__ == "__main__":
    main()