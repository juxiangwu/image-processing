#coding:utf-8

from pyglet.gl import *

import OpenGL.GL.shaders
import ctypes
import pyrr
import time
from math import sin

class Triangle:
    def __init__(self):

        self.triangle = [-0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
                          0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
                          0.0,  0.5, 0.0, 0.0, 0.0, 1.0]

        self.vertex_shader_source = b"""
        #version 440
        in layout(location = 0) vec3 position;
        in layout(location = 1) vec3 color;
        uniform mat4 scale;
        uniform mat4 rotate;
        uniform mat4 translate;
        out vec3 newColor;
        void main()
        {
            gl_Position = translate * rotate * scale * vec4(position, 1.0f);
            newColor = color;
        }
        """

        self.fragment_shader_source = b"""
        #version 440
        in vec3 newColor;
        out vec4 outColor;
        void main()
        {
            outColor = vec4(newColor, 1.0f);
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
        glBufferData(GL_ARRAY_BUFFER, 72, (GLfloat * len(self.triangle))(*self.triangle), GL_STATIC_DRAW)

        #positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        #colors
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        self.scale_loc = glGetUniformLocation(shader, b'scale')
        self.rotate_loc = glGetUniformLocation(shader, b'rotate')
        self.translate_loc = glGetUniformLocation(shader, b'translate')

        self.scale = pyrr.Matrix44.identity()
        self.rot_y = pyrr.Matrix44.identity()
        self.translate = pyrr.Matrix44.identity()

    def transform(self):
        ct = time.clock()
        self.scale = pyrr.Matrix44.from_scale(pyrr.Vector3([abs(sin(ct)), abs(sin(ct)), 1.0])).flatten()
        self.rot_y = pyrr.Matrix44.from_y_rotation(ct*2).flatten()
        self.translate = pyrr.Matrix44.from_translation(pyrr.Vector3([sin(ct), sin(ct*0.5), 0.0])).flatten()

        c_scale = (GLfloat * len(self.scale))(*self.scale)
        c_rotate_y = (GLfloat * len(self.rot_y))(*self.rot_y)
        c_translate = (GLfloat * len(self.translate))(*self.translate)

        glUniformMatrix4fv(self.scale_loc, 1, GL_FALSE, c_scale)
        glUniformMatrix4fv(self.rotate_loc, 1, GL_FALSE, c_rotate_y)
        glUniformMatrix4fv(self.translate_loc, 1, GL_FALSE, c_translate)

class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)
        glClearColor(0.2, 0.3, 0.2, 1.0)

        self.triangle = Triangle()


    def on_draw(self):
        self.clear()
        glDrawArrays(GL_TRIANGLES, 0, 3)

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)

    def update(self, dt):
        self.triangle.transform()


if __name__ == "__main__":
    window = MyWindow(800, 600, "My Pyglet Window", resizable=True)
    pyglet.clock.schedule_interval(window.update, 1/30.0)
    pyglet.app.run()
