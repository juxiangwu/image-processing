#coding:utf-8

import sys
from OpenGL.GL import *
from OpenGL.GLUT import *

def main():
    glutInit(sys.argv)

    # Create a double-buffer RGBA window.   (Single-buffering is possible.
    # So is creating an index-mode window.)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

    # Create a window, setting its title
    glutCreateWindow(b'Hello GLUT')

    # Set the display callback.  You can set other callbacks for keyboard and
    # mouse events.
    glutDisplayFunc(onDisplay)
    glutKeyboardFunc(onKeyEvent)

    printSystemGLInfo()

    # Run the GLUT main loop until the user closes the window.
    glutMainLoop()


def printSystemGLInfo():
    print('Vendor: %s' % (glGetString(GL_VENDOR)).decode("utf-8") )
    print('Opengl version: %s' % (glGetString(GL_VERSION).decode("utf-8") ))
    print('GLSL Version: %s' % (glGetString(GL_SHADING_LANGUAGE_VERSION)).decode("utf-8") )
    print('Renderer: %s' % (glGetString(GL_RENDERER)).decode("utf-8") )

# The display() method does all the work; it has to call the appropriate
# OpenGL functions to actually display something.
def onDisplay():
    # Clear the color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # ... render stuff in here ...
    # It will go to an off-screen frame buffer.

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