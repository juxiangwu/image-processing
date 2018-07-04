#coding:utf-8

import sys
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import compileShader, compileProgram
from OpenGL.GL.ARB import vertex_array_object
from os.path import dirname, join
import numpy as np
import math

# Number of bytes in a GLFloat
sizeOfFloat = 4
sizeOfUshort = 2

class GameObject:
    def __init__(self, vertices, colours, indices):
        self.vertexPositions = vertices
        self.vertexComponents = 4
        self.vertexColours = colours
        self.colourComponents = 4
        self.indices = indices
        self.vertexArrayObject = glGenVertexArrays(1)
        self.positionBufferObject, self.colourBufferObject = self.initialize_vertex_buffers()
        self.indexBufferObject = self.initialize_index_buffer()
        self.model = np.identity(4, dtype='float32')

    def initialize_vertex_buffers(self):
        glBindVertexArray(self.vertexArrayObject)
        positionBufferObject = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject)
        array_type = (GLfloat * len(self.vertexPositions))
        glBufferData(
            GL_ARRAY_BUFFER, len(self.vertexPositions) * sizeOfFloat,
            array_type(*self.vertexPositions), GL_STATIC_DRAW
        )
        colourBufferObject = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, colourBufferObject)
        array_type = (GLfloat * len(self.vertexColours))
        glBufferData(
            GL_ARRAY_BUFFER, len(self.vertexColours) * sizeOfFloat,
            array_type(*self.vertexColours), GL_STATIC_DRAW
        )
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        return positionBufferObject, colourBufferObject

    def initialize_index_buffer(self):
        glBindVertexArray(self.vertexArrayObject)
        indexBufferObject = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferObject)
        array_type = (GLushort * len(self.indices))
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER, len(self.indices) * sizeOfUshort,
            array_type(*self.indices), GL_STATIC_DRAW
        )
        glBindVertexArray(0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        return indexBufferObject

class ShaderProgram:
    def __init__(self, gameObjects, vShaderFilename, fShaderFilename):
        self.gameObjects = gameObjects
        glBindVertexArray(gameObjects[0].vertexArrayObject)
        self.shaderProgram = compileProgram(
            compileShader(
                loadFile(vShaderFilename), GL_VERTEX_SHADER),
            compileShader(
                loadFile(fShaderFilename), GL_FRAGMENT_SHADER)
        )
        glBindVertexArray(0)
        self.setup_attributes()

    def setup_attributes(self):
        for go in self.gameObjects:
            glBindVertexArray(go.vertexArrayObject)
            glBindBuffer(GL_ARRAY_BUFFER, go.positionBufferObject)
            positionLocation= glGetAttribLocation(self.shaderProgram, b'position')
            glEnableVertexAttribArray(positionLocation)
            glVertexAttribPointer(positionLocation, go.vertexComponents, GL_FLOAT, False, 0, None)
            glBindBuffer(GL_ARRAY_BUFFER, go.colourBufferObject)
            colourLocation = glGetAttribLocation(self.shaderProgram, b'colour_in')
            glEnableVertexAttribArray(colourLocation)
            glVertexAttribPointer(colourLocation, go.colourComponents, GL_FLOAT, False, 0, None)
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    def drawGameObjects(self, view, proj):
        glUseProgram(self.shaderProgram)
        for go in self.gameObjects:
            glBindVertexArray(go.vertexArrayObject)
            uniformLocation = glGetUniformLocation(self.shaderProgram, b'model')
            if uniformLocation != -1:
                glUniformMatrix4fv(uniformLocation, 1, False, go.model.tolist())
            else:
                print("Error: model location not found")
            uniformLocation = glGetUniformLocation(self.shaderProgram, b'view')
            if uniformLocation != -1:
                glUniformMatrix4fv(uniformLocation, 1, False, view.tolist())
            else:
                print("Error: view location not found")
            uniformLocation = glGetUniformLocation(self.shaderProgram, b'proj')
            if uniformLocation != -1:
                glUniformMatrix4fv(uniformLocation, 1, False, proj.tolist())
            else:
                print("Error: proj location not found")
            glDrawElements(GL_TRIANGLES, len(go.indices), GL_UNSIGNED_SHORT, None)
            glBindVertexArray(0)
        glUseProgram(0)

def main():
    global triangleShader
    global projMatrix
    global viewMatrix

    global fieldOfView
    global nearClipPlane
    global farClipPlane

    fieldOfView = 60.0
    nearClipPlane = 0.1
    farClipPlane = 100.0
    screenWidth = 720
    screenHeight = 480

    glutInit(sys.argv)

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(screenWidth, screenHeight)

    glutCreateWindow(b'Hello MVP Matrix')

    glutReshapeFunc(reshapeWindow)
    glutDisplayFunc(display)
    glutKeyboardFunc(onKeyEvent)

    printSystemGLInfo()

    glClearColor (0.1, 0.2, 0.3, 1.0)

    # Three vertices, with an x,y,z & w for each.
    vertexPositions = [
        0.0, 0.0,  0.0,  1.0,
        0.5, 0.0,  0.0,  1.0,
        0.0, 0.5,  0.0,  1.0,
    ]

    vertexColours = [
        0.0, 0.8, 0.0, 1.0,
        0.0, 0.5, 0.0, 1.0,
        0.0, 0.2, 0.0, 1.0,
    ]

    indices = [
        0,1,2
    ]
    
    triangle = GameObject(vertexPositions, vertexColours, indices)
    triangleShader = ShaderProgram([triangle], '../../datas/glsl/simpleMVP.vert', '../../datas/glsl/simpleMVP.frag')

    projMatrix = perspective(fieldOfView, float(screenWidth)/float(screenHeight), nearClipPlane, farClipPlane)
    viewMatrix = lookAt(np.array([0.0,0.0,1.0]), np.array([0.0,0.0,0.0]), np.array([0.0,1.0,0.0]))

    # Run the GLUT main loop until the user closes the window.
    glutMainLoop()


def printSystemGLInfo():
    print('Vendor: %s' % (glGetString(GL_VENDOR)).decode("utf-8") )
    print('Opengl version: %s' % (glGetString(GL_VERSION).decode("utf-8") ))
    print('GLSL Version: %s' % (glGetString(GL_SHADING_LANGUAGE_VERSION)).decode("utf-8") )
    print('Renderer: %s' % (glGetString(GL_RENDERER)).decode("utf-8") )

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    triangleShader.drawGameObjects(viewMatrix, projMatrix)
    glutSwapBuffers()

def reshapeWindow(width, height):
    global projMatrix
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    projMatrix = perspective(fieldOfView, float(width)/float(height), nearClipPlane, farClipPlane)

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

# https://github.com/tartley/gltutpy/blob/master/t02.playing-with-colors/glwrap.py
def glGenVertexArray():
    vao_id = GLuint(0)
    vertex_array_object.glGenVertexArrays(1, vao_id)
    return vao_id.value

# Modified from
# https://stackoverflow.com/questions/35369483/opengl-perspective-matrix-in-python
def perspective(field_of_view_y, aspect, z_near, z_far):
    fov_radians = math.radians(field_of_view_y)
    f = math.tan(fov_radians/2.0)

    a_11 = 1.0/(f*aspect)
    a_22 = 1.0/f
    a_33 = -(z_near + z_far)/(z_near - z_far)
    a_34 = -2.0*z_near*z_far/(z_near - z_far)

    perspective_matrix = np.matrix([
        [a_11, 0, 0, 0],       
        [0, a_22, 0, 0],       
        [0, 0, a_33, a_34],    
        [0, 0, -1, 0]          
    ], dtype='float32').T 

    return perspective_matrix

# Modified from
# https://github.com/mackst/glm/blob/f532befb0412f96938fe672bd3fc02cd5b3e74b5/glm/gtc/matrix_transform.py
def lookAt(eye, center, up):
    f = normalized(center - eye)[0,:]
    s = normalized(np.cross(f, up))[0,:]
    u = np.cross(s, f)

    view = np.identity(4, dtype='float32')
    view[0][0] = s[0]
    view[1][0] = s[1]
    view[2][0] = s[2]
    view[0][1] = u[0]
    view[1][1] = u[1]
    view[2][1] = u[2]
    view[0][2] =-f[0]
    view[1][2] =-f[1]
    view[2][2] =-f[2]
    view[3][0] =-np.dot(s, eye)
    view[3][1] =-np.dot(u, eye)
    view[3][2] = np.dot(f, eye)
    return view

# https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
def normalized(a, axis=-1, order=1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

if __name__ == "__main__":
    main()