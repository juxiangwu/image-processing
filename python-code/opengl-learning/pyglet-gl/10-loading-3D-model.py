#coding:utf-8

from pyglet.gl import *
import numpy as np
import OpenGL.GL.shaders
import ctypes
import pyrr
import time
from math import sin

class ObjLoader:
    def __init__(self):
        self.vert_coords = []
        self.text_coords = []
        self.norm_coords = []

        self.vertex_index = []
        self.texture_index = []
        self.normal_index = []

        self.model = []
        self.c_model = None

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
            self.model.extend(self.vert_coords[i])

        for i in self.texture_index:
            self.model.extend(self.text_coords[i])

        for i in self.normal_index:
            self.model.extend(self.norm_coords[i])

        self.model = np.array(self.model, dtype='float32')

        self.c_model = (GLfloat * len(self.model))(*self.model)

class Model:
    def __init__(self):

        self.tree = ObjLoader()
        self.tree.load_model("datas/models/opengl/xmas_tree.obj")

        self.vertex_shader_source = b"""
        #version 440
        in layout(location = 0) vec3 positions;
        in layout(location = 1) vec2 textureCoords;
        in layout(location = 2) vec3 normals;
        uniform mat4 light;
        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;
        uniform mat4 rotate;
        out vec2 textures;
        out vec3 fragNormal;
        void main()
        {
            fragNormal = (light * vec4(normals, 0.0f)).xyz;
            gl_Position =  projection * view * model * rotate * vec4(positions, 1.0f);
            textures = vec2(textureCoords.x, 1 - textureCoords.y);
        }
        """

        self.fragment_shader_source = b"""
        #version 440
        in vec2 textures;
        in vec3 fragNormal;
        uniform sampler2D sampTexture;
        out vec4 outColor;
        void main()
        {
            vec3 ambientLightIntensity = vec3(0.3f, 0.2f, 0.4f);
            vec3 sunLightIntensity = vec3(0.9f, 0.9f, 0.9f);
            vec3 sunLightDirection = normalize(vec3(1.0f, 1.0f, -0.5f));
            vec4 texel = texture(sampTexture, textures);
            vec3 lightIntensity = ambientLightIntensity + sunLightIntensity * max(dot(fragNormal, sunLightDirection), 0.0f);
            outColor = vec4(texel.rgb * lightIntensity, texel.a);
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
        glBufferData(GL_ARRAY_BUFFER, len(self.tree.model)*4, self.tree.c_model, GL_STATIC_DRAW)

        texture_offset = len(self.tree.vertex_index) * 12
        normal_offset = (texture_offset + len(self.tree.texture_index) * 8)

        #positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.tree.model.itemsize * 3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        #textures
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.tree.model.itemsize * 2, ctypes.c_void_p(texture_offset))
        glEnableVertexAttribArray(1)

        #normals
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.tree.model.itemsize * 3, ctypes.c_void_p(normal_offset))
        glEnableVertexAttribArray(2)

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

        xmas = pyglet.image.load('datas/models/opengl/xmas_texture.jpg')
        image_data = xmas.get_data('RGB', xmas.pitch)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, xmas.width, xmas.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)

        view = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, -3.0])).flatten()
        projection = pyrr.matrix44.create_perspective_projection_matrix(62.0, 1280 / 720, 0.1, 100.0).flatten()
        model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.5, -2.0])).flatten()

        c_view = (GLfloat * len(view))(*view)
        c_projection = (GLfloat * len(projection))(*projection)
        c_model = (GLfloat * len(model))(*model)

        self.rotate_loc = glGetUniformLocation(shader, b'rotate')
        self.view_loc = glGetUniformLocation(shader, b"view")
        self.proj_loc = glGetUniformLocation(shader, b"projection")
        self.model_loc = glGetUniformLocation(shader, b"model")
        self.light_loc = glGetUniformLocation(shader, b"light")

        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, c_view)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, c_projection)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, c_model)

        self.rot_y = pyrr.Matrix44.identity()

    def rotate(self):

        ct = time.clock()
        self.rot_y = pyrr.Matrix44.from_y_rotation(ct).flatten()

        c_rotate = (GLfloat * len(self.rot_y))(*self.rot_y)

        glUniformMatrix4fv(self.rotate_loc, 1, GL_FALSE, c_rotate)
        glUniformMatrix4fv(self.light_loc, 1, GL_FALSE, c_rotate)

class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)
        glClearColor(0.2, 0.3, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)

        self.model = Model()

    def on_draw(self):
        self.clear()
        glDrawArrays(GL_TRIANGLES, 0, len(self.model.tree.vertex_index))

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)

    def update(self, dt):
        self.model.rotate()

if __name__ == "__main__":
    window = MyWindow(800, 600, "My Pyglet Window", resizable=True)
    pyglet.clock.schedule_interval(window.update, 1/30.0)
    pyglet.app.run()