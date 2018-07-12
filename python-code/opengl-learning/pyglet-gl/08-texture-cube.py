#coding:utf-8

from pyglet.gl import *

import OpenGL.GL.shaders
import ctypes
import pyrr
import time
from math import sin

class Cube:
    def __init__(self):
        self.cube = [-0.5, -0.5, 0.5, 0.0, 0.0,
                      0.5, -0.5, 0.5, 1.0, 0.0,
                      0.5,  0.5, 0.5, 1.0, 1.0,
                     -0.5,  0.5, 0.5, 0.0, 1.0,

                     -0.5, -0.5, -0.5, 0.0, 0.0,
                      0.5, -0.5, -0.5, 1.0, 0.0,
                      0.5,  0.5, -0.5, 1.0, 1.0,
                     -0.5,  0.5, -0.5, 0.0, 1.0,

                      0.5, -0.5, -0.5, 0.0, 0.0,
                      0.5,  0.5, -0.5, 1.0, 0.0,
                      0.5,  0.5,  0.5, 1.0, 1.0,
                      0.5, -0.5,  0.5, 0.0, 1.0,

                     -0.5,  0.5, -0.5, 0.0, 0.0,
                     -0.5, -0.5, -0.5, 1.0, 0.0,
                     -0.5, -0.5,  0.5, 1.0, 1.0,
                     -0.5,  0.5,  0.5, 0.0, 1.0,

                     -0.5, -0.5, -0.5, 0.0, 0.0,
                      0.5, -0.5, -0.5, 1.0, 0.0,
                      0.5, -0.5,  0.5, 1.0, 1.0,
                     -0.5, -0.5,  0.5, 0.0, 1.0,

                      0.5, 0.5, -0.5, 0.0, 0.0,
                     -0.5, 0.5, -0.5, 1.0, 0.0,
                     -0.5, 0.5,  0.5, 1.0, 1.0,
                      0.5, 0.5,  0.5, 0.0, 1.0]

        self.indices = [0,  1,  2,  2,  3,  0,
                        4,  5,  6,  6,  7,  4,
                        8,  9, 10, 10, 11,  8,
                       12, 13, 14, 14, 15, 12,
                       16, 17, 18, 18, 19, 16,
                       20, 21, 22, 22, 23, 20]

        self.vertex_shader_source = b"""
        #version 440
        in layout(location = 0) vec3 position;
        in layout(location = 1) vec2 textureCoords;
        uniform mat4 rotate;
        out vec2 textures;
        void main()
        {
            gl_Position = rotate * vec4(position, 1.0f);
            textures = textureCoords;
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
        glBufferData(GL_ARRAY_BUFFER, len(self.cube) * 4, (GLfloat * len(self.cube))(*self.cube), GL_STATIC_DRAW)

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

        crate = pyglet.image.load('datas/smile_mask.jpg')
        image_data = crate.get_data('RGB', crate.pitch)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, crate.width, crate.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)

        self.rotate_loc = glGetUniformLocation(shader, b'rotate')

        self.rot_x = pyrr.Matrix44.identity()
        self.rot_y = pyrr.Matrix44.identity()

    def rotate(self):
        ct = time.clock()
        self.rot_x = pyrr.Matrix44.from_x_rotation(ct)
        self.rot_y = pyrr.Matrix44.from_y_rotation(ct)

        rotate = (self.rot_x * self.rot_y).flatten()

        c_rotate = (GLfloat * len(rotate))(*rotate)

        glUniformMatrix4fv(self.rotate_loc, 1, GL_FALSE, c_rotate)


class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)
        glClearColor(0.2, 0.3, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)

        self.cube = Cube()


    def on_draw(self):
        self.clear()
        #glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glDrawElements(GL_TRIANGLES, len(self.cube.indices), GL_UNSIGNED_INT, None)


    def on_resize(self, width, height):
        glViewport(0, 0, width, height)

    def update(self, dt):
        self.cube.rotate()


if __name__ == "__main__":
    window = MyWindow(800, 600, "My Pyglet Window", resizable=True)
    pyglet.clock.schedule_interval(window.update, 1/30.0)
    pyglet.app.run()
