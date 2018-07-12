#coding:utf-8

from pyglet.gl import *

import OpenGL.GL.shaders
import ctypes

class Triangle:
    def __init__(self):

        self.triangle = [-0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
                          0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
                          0.0,  0.5, 0.0, 0.0, 0.0, 1.0]

        self.vertex_shader_source = b"""
        #version 440
        in layout(location = 0) vec3 position;
        in layout(location = 1) vec3 color;
        out vec3 newColor;
        void main()
        {
            gl_Position = vec4(position, 1.0f);
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


if __name__ == "__main__":
    window = MyWindow(800, 600, "My Pyglet Window", resizable=True)
    pyglet.app.run()