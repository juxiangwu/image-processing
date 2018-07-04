#coding:utf-8

import sys
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import compileShader, compileProgram
from OpenGL.GL.ARB import vertex_array_object
from os.path import dirname, join

# Number of bytes in a GLFloat
sizeOfFloat = 4
sizeOfUshort = 2

# Three vertices, with an x,y,z & w for each.
vertexPositions = [
     0.0, 0.0,  0.0,  1.0,
     0.5, 0.0,  0.0,  1.0,
     0.0, 0.5,  0.0,  1.0,
]
vertexComponents = 4

vertexColours = [
    0.0, 0.8, 0.0, 1.0,
    0.0, 0.5, 0.0, 1.0,
    0.0, 0.2, 0.0, 1.0,
]
colourComponents = 4

indices = [
    0,1,2
]

def main():
    glutInit(sys.argv)

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(720, 480)

    glutCreateWindow(b'Modern Hello Triangle with VBO')

    glutDisplayFunc(display)
    glutKeyboardFunc(onKeyEvent)

    printSystemGLInfo()

    glClearColor (0.1, 0.2, 0.3, 1.0)

    initialize_vertex_array()
    initialize_program()
    setup_attributes()
    # Run the GLUT main loop until the user closes the window.
    glutMainLoop()


def printSystemGLInfo():
    print('Vendor: %s' % (glGetString(GL_VENDOR)).decode("utf-8") )
    print('Opengl version: %s' % (glGetString(GL_VERSION).decode("utf-8") ))
    print('GLSL Version: %s' % (glGetString(GL_SHADING_LANGUAGE_VERSION)).decode("utf-8") )
    print('Renderer: %s' % (glGetString(GL_RENDERER)).decode("utf-8") )

def initialize_program():
    global shaderProgram
    glBindVertexArray(vertexArrayObject)
    shaderProgram = compileProgram(
        compileShader(
            loadFile('../../datas/glsl/simpleVAO.vert'), GL_VERTEX_SHADER),
        compileShader(
            loadFile('../../datas/glsl/simpleVAO.frag'), GL_FRAGMENT_SHADER)
    )
    glBindVertexArray(0)

def initialize_vertex_array():
    global vertexArrayObject
    vertexArrayObject = glGenVertexArrays(1)
    glBindVertexArray(vertexArrayObject)
    initialize_vertex_buffer()
    initialize_index_buffer()
    glBindVertexArray(0)

def initialize_vertex_buffer():
    global positionBufferObject
    positionBufferObject = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject)
    array_type = (GLfloat * len(vertexPositions))
    glBufferData(
        GL_ARRAY_BUFFER, len(vertexPositions) * sizeOfFloat,
        array_type(*vertexPositions), GL_STATIC_DRAW
    )

    global colourBufferObject
    colourBufferObject = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, colourBufferObject)
    array_type = (GLfloat * len(vertexColours))
    glBufferData(
        GL_ARRAY_BUFFER, len(vertexColours) * sizeOfFloat,
        array_type(*vertexColours), GL_STATIC_DRAW
    )

def initialize_index_buffer():
    global indexBufferObject
    indexBufferObject = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferObject)
    array_type = (GLushort * len(indices))
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER, len(indices) * sizeOfUshort,
        array_type(*indices), GL_STATIC_DRAW
    )
    
def setup_attributes():
    glBindVertexArray(vertexArrayObject)
    glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject)
    positionLocation= glGetAttribLocation(shaderProgram, b'position')
    glEnableVertexAttribArray(positionLocation)
    glVertexAttribPointer(positionLocation, vertexComponents, GL_FLOAT, False, 0, None)
    glBindBuffer(GL_ARRAY_BUFFER, colourBufferObject)
    colourLocation = glGetAttribLocation(shaderProgram, b'colour_in')
    glEnableVertexAttribArray(colourLocation)
    glVertexAttribPointer(colourLocation, colourComponents, GL_FLOAT, False, 0, None)
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shaderProgram)
    glBindVertexArray(vertexArrayObject)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_SHORT, None)
    glBindVertexArray(0)
    glUseProgram(0)

    glutSwapBuffers()

# https://butterflyofdream.wordpress.com/2016/04/27/pyopengl-keyboard-wont-respond/
def onKeyEvent(bkey, x, y):
    # Convert bytes object to string 
    key = bkey.decode("utf-8")
    # Allow to quit by pressing 'Esc'
    if key == chr(27):
        print("Exiting")
        sys.exit()

def loadFile(filename):
    with open(join(dirname(__file__), filename)) as fp:
        return fp.read()

# 
# https://github.com/tartley/gltutpy/blob/master/t02.playing-with-colors/glwrap.py
def glGenVertexArray():
    vao_id = GLuint(0)
    vertex_array_object.glGenVertexArrays(1, vao_id)
    return vao_id.value

if __name__ == "__main__":
    main()