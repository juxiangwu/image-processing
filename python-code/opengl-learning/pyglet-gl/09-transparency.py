#coding:utf-8

from pyglet.gl import *

import OpenGL.GL.shaders
import ctypes
import pyrr
import time
from math import sin

class Quad:
    def __init__(self):
        self.quad = [-0.5, -0.5, 0.0, 0.0, 0.0,
                      0.5, -0.5, 0.0, 1.0, 0.0,
                      0.5,  0.5, 0.0, 1.0, 1.0,
                     -0.5,  0.5, 0.0, 0.0, 1.0]

        self.indices = [0,  1,  2,  2,  3,  0]

        self.vertex_shader_source = b"""
        #version 440
        in layout(location = 0) vec3 position;
        in layout(location = 1) vec2 textureCoords;
        uniform mat4 rotate;
        out vec2 textures;
        void main()
        {
            gl_Position = rotate * vec4(position, 1.0f);
            textures = vec2(textureCoords.x, 1 - textureCoords.y);
        }
        """

        self.fragment_shader_source = b"""
        #version 440
        in vec2 textures;
        uniform sampler2D sampTexture;
        out vec4 outColor;
        void main()
        {
            outColor = texture(sampTexture, textures);
        }
        """

        vertex_buff = ctypes.create_string_buffer(self.vertex_shader_source)
        c_vertex = ctypes.cast(ctypes.pointer(ctypes.pointer(vertex_buff)), ctypes.POINTER(ctypes.POINTER(GLchar)))
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, 1, c_vertex, None)
        glCompileShader(vertex_shader)

        fragment_buff = ctypes.create_string_buffer(self.fragment_shader_source)
        c_fragment = ctypes.cast(ctypes.pointer(ctypes.pointer(fragment_buff)), ctypes.POINTER(ctypes.POINTER(GLchar)))
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, 1, c_fragment, None)
        glCompileShader(fragment_shader)

        shader = glCreateProgram()
        glAttachShader(shader, vertex_shader)
        glAttachShader(shader, fragment_shader)
        glLinkProgram(shader)

        glUseProgram(shader)

        vbo = GLuint(0)
        glGenBuffers(1, vbo)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, len(self.quad) * 4, (GLfloat * len(self.quad))(*self.quad), GL_STATIC_DRAW)

        ebo = GLuint(0)
        glGenBuffers(1, ebo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self.indices)*4, (GLuint * len(self.indices))(*self.indices), GL_STATIC_DRAW)

        #positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        #textures
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        texture = GLuint(0)
        glGenTextures(1, texture)
        glBindTexture(GL_TEXTURE_2D, texture)
        #set the texture wrapping
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        #set the texture filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        image = pyglet.image.load('datas/cat2.png')
        image_data = image.get_data('RGBA', image.pitch) #image.width*4
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)

        self.rotate_loc = glGetUniformLocation(shader, b'rotate')

        self.rot_y = pyrr.Matrix44.identity()

    def rotate(self):

        ct = time.clock()
        self.rot_y = pyrr.Matrix44.from_y_rotation(ct).flatten()

        c_rotate = (GLfloat * len(self.rot_y))(*self.rot_y)

        glUniformMatrix4fv(self.rotate_loc, 1, GL_FALSE, c_rotate)

class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)
        glClearColor(0.2, 0.3, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)

        self.quad = Quad()


    def on_draw(self):
        self.clear()
        glDrawElements(GL_TRIANGLES, len(self.quad.indices), GL_UNSIGNED_INT, None)


    def on_resize(self, width, height):
        glViewport(0, 0, width, height)

    def update(self, dt):
        self.quad.rotate()


if __name__ == "__main__":
    window = MyWindow(800, 600, "My Pyglet Window", resizable=True)
    pyglet.clock.schedule_interval(window.update, 1/30.0)
    pyglet.app.run()