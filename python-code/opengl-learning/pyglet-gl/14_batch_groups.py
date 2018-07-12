#coding:utf-8

from pyglet.gl import *
import numpy as np
import OpenGL.GL.shaders
import ctypes
import pyrr
import time
from math import sin
from pyrr import Vector3, matrix44, Matrix44
import time
import numpy

class ObjLoader:
    def __init__(self):
        self.vert_coords = []
        self.text_coords = []
        self.norm_coords = []

        self.vertex_index = []
        self.texture_index = []
        self.normal_index = []

        self.model_vertices = []
        self.model_textures = []
        self.model_normals = []

    def load_model(self, file):
        for line in open(file, 'r'):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue

            if values[0] == 'v':
                self.vert_coords.append(values[1:4])
            if values[0] == 'vt':
                self.text_coords.append(values[1:3])
            if values[0] == 'vn':
                self.norm_coords.append(values[1:4])

            if values[0] == 'f':
                face_i = []
                text_i = []
                norm_i = []
                for v in values[1:4]:
                    w = v.split('/')
                    face_i.append(int(w[0])-1)
                    text_i.append(int(w[1])-1)
                    norm_i.append(int(w[2])-1)
                self.vertex_index.append(face_i)
                self.texture_index.append(text_i)
                self.normal_index.append(norm_i)

        self.vertex_index = [y for x in self.vertex_index for y in x]
        self.texture_index = [y for x in self.texture_index for y in x]
        self.normal_index = [y for x in self.normal_index for y in x]

        for i in self.vertex_index:
            self.model_vertices.extend(map(float, self.vert_coords[i]))

        for i in self.texture_index:
            self.model_textures.extend(map(float, self.text_coords[i]))

        for i in self.normal_index:
            self.model_normals.extend(map(float, self.norm_coords[i]))


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

main_batch = pyglet.graphics.Batch()


class Shader:
    model_loc = None

    @staticmethod
    def init():
        shader = compile_shader("datas/glsl/video_13_vert.glsl", "datas/glsl/video_13_frag.glsl")
        glUseProgram(shader)

        view = matrix44.create_from_translation(Vector3([0.0, 0.0, -2.0])).flatten().astype("float32")
        projection = matrix44.create_perspective_projection_matrix(45.0, 1280 / 720, 0.1, 100.0).flatten().astype("float32")

        c_view = numpy.ctypeslib.as_ctypes(view)
        c_projection = numpy.ctypeslib.as_ctypes(projection)

        view_loc = glGetUniformLocation(shader, b"view")
        proj_loc = glGetUniformLocation(shader, b"projection")
        Shader.model_loc = glGetUniformLocation(shader, b"model")

        glUniformMatrix4fv(view_loc, 1, GL_FALSE, c_view)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, c_projection)


class Monkey:
    def __init__(self):

        mesh = ObjLoader()
        mesh.load_model("datas/models/opengl/monkey.obj")

        num_verts = len(mesh.model_vertices) // 3

        group = pyglet.graphics.Group()
        group.set_state = self.state

        self.verts = main_batch.add(num_verts, GL_TRIANGLES, group, ('v3f', mesh.model_vertices),
                                                                    ('t2f', mesh.model_textures))

        self.model = matrix44.create_from_translation(Vector3([-2.0, 0.0, -4.0])).flatten().astype("float32")
        self.c_model = numpy.ctypeslib.as_ctypes(self.model)

        # region texture settings
        self.texture = GLuint(0)
        glGenTextures(1, self.texture)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        # set the texture wrapping
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # set the texture filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        image = pyglet.image.load('datas/models/opengl/monkey.jpg')
        image_data = image.get_data('RGB', image.pitch)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
        # endregion

    def state(self):
        # vertices
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, self.verts.vertices)
        # textures
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, self.verts.tex_coords)

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniformMatrix4fv(Shader.model_loc, 1, GL_FALSE, self.c_model)


class Cube:
    def __init__(self):
        mesh = ObjLoader()
        mesh.load_model("datas/models/opengl/cube.obj")

        num_verts = len(mesh.model_vertices) // 3

        group = pyglet.graphics.Group()
        group.set_state = self.state

        self.verts = main_batch.add(num_verts, GL_TRIANGLES, group, ('v3f', mesh.model_vertices),
                                                                    ('t2f', mesh.model_textures))

        self.model = matrix44.create_from_translation(Vector3([2.0, 0.0, -4.0])).flatten().astype("float32")
        self.c_model = numpy.ctypeslib.as_ctypes(self.model)

        # region texture settings
        self.texture = GLuint(0)
        glGenTextures(1, self.texture)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        # set the texture wrapping
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # set the texture filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        image = pyglet.image.load('datas/models/opengl/cube.jpg')
        image_data = image.get_data('RGB', image.pitch)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
        # endregion

    def state(self):
        # vertices
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, self.verts.vertices)
        # textures
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, self.verts.tex_coords)

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniformMatrix4fv(Shader.model_loc, 1, GL_FALSE, self.c_model)


class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)
        glClearColor(0.2, 0.3, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)

        Shader.init()
        self.cube = Cube()
        self.monkey = Monkey()

    def on_draw(self):
        self.clear()
        main_batch.draw()

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)

    def update(self, dt):
        pass


if __name__ == "__main__":
    window = MyWindow(800, 600, "My Pyglet Window", resizable=True)
    pyglet.clock.schedule_interval(window.update, 1/60.0)
    pyglet.app.run()