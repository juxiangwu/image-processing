#coding:utf-8

from pyglet.gl import *
import numpy as np
import OpenGL.GL.shaders
import ctypes
import pyrr
import time
from math import sin

def load_shader(shader_file):
    with open(shader_file) as f:
        shader_source = f.read()
    return str.encode(shader_source)


def compile_shader(vs, fs):
    vert_shader = load_shader(vs)
    frag_shader = load_shader(fs)

    vertex_buff = ctypes.create_string_buffer(vert_shader)
    c_vertex = ctypes.cast(ctypes.pointer(ctypes.pointer(vertex_buff)), ctypes.POINTER(ctypes.POINTER(GLchar)))
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, 1, c_vertex, None)
    glCompileShader(vertex_shader)

    fragment_buff = ctypes.create_string_buffer(frag_shader)
    c_fragment = ctypes.cast(ctypes.pointer(ctypes.pointer(fragment_buff)), ctypes.POINTER(ctypes.POINTER(GLchar)))
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, 1, c_fragment, None)
    glCompileShader(fragment_shader)

    shader = glCreateProgram()
    glAttachShader(shader, vertex_shader)
    glAttachShader(shader, fragment_shader)
    glLinkProgram(shader)

    return shader

class Triangle:
    def __init__(self):
        self.verts = pyglet.graphics.vertex_list(3, ('v3f', (-0.5,-0.5,0.0, 0.5,-0.5,0.0, 0.0,0.5,0.0)),
                                                    ('c3f', ( 1.0, 0.0,0.0, 0.0, 1.0,0.0, 0.0,0.0,1.0)))

        shader = compile_shader("datas/glsl/video_11_vert.glsl", "datas/glsl/video_11_frag.glsl")

        glUseProgram(shader)

        # vertices
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, self.verts.vertices)
        glEnableVertexAttribArray(0)

        # colors
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, self.verts.colors)
        glEnableVertexAttribArray(1)



class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)
        glClearColor(0.2, 0.3, 0.2, 1.0)

        self.triangle = Triangle()

    def on_draw(self):
        self.clear()
        self.triangle.verts.draw(GL_TRIANGLES)

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)

    def update(self, dt):
        pass


if __name__ == "__main__":
    window = MyWindow(800, 600, "My Pyglet Window", resizable=True)
    pyglet.clock.schedule_interval(window.update, 1/60.0)
    pyglet.app.run()