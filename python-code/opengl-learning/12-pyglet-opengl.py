#coding:utf-8
import pyglet
from pyglet.gl import *
import ctypes as cts
frag = '''
uniform float time;
uniform float surface_y;
uniform float wave_height;
uniform float wave_length;
uniform float wave_speed;

void main(void)
{
    float pi = 3.14159265358979323846264;
    vec2 tex_coord = vec2(gl_TexCoord[0]);
    float x = tex_coord.x;
    float y = tex_coord.y;
    if (y < surface_y + 0.5 * wave_height) {
        if (y < surface_y - 0.5 * wave_height) {
            gl_FragColor = gl_Color;
        } else if (y < surface_y + 0.2 * wave_height -
                   0.7 * wave_height * abs(sin(2.0 * pi * -wave_speed * time + 2.0 * pi * x / wave_length)) +
                   0.3 * wave_height * pow(sin(4.0 * pi * -wave_speed * time + pi * x / wave_length), 2.0))
        {
            gl_FragColor = gl_Color;
        }
    }
}
'''

class Shader:
	# vert, frag and geom take arrays of source strings
	# the arrays will be concattenated into one string by OpenGL
	def __init__(self, vert = [], frag = [], geom = []):
		# create the program handle
		self.handle = glCreateProgram()
		# we are not linked yet
		self.linked = False

		# create the vertex shader
		self.createShader(vert, GL_VERTEX_SHADER)
		# create the fragment shader
		self.createShader(frag, GL_FRAGMENT_SHADER)
		# the geometry shader will be the same, once pyglet supports the extension
		# self.createShader(frag, GL_GEOMETRY_SHADER_EXT)

		# attempt to link the program
		self.link()

	def createShader(self, strings, type):
        
		count = len(strings)
		# if we have no source code, ignore this shader
		if count < 1:
			return

		# create the shader handle
		shader = glCreateShader(type)

		# convert the source strings into a ctypes pointer-to-char array, and upload them
		# this is deep, dark, dangerous black magick - don't try stuff like this at home!
		src = (cts.c_char_p * count)(*strings)
		glShaderSource(shader, count, cts.cast(pointer(src), cts.POINTER(cts.POINTER(cts.c_char))), None)

		# compile the shader
		glCompileShader(shader)

		temp = cts.c_int(0)
		# retrieve the compile status
		glGetShaderiv(shader, GL_COMPILE_STATUS, byref(temp))

		# if compilation failed, print the log
		if not temp:
			# retrieve the log length
			glGetShaderiv(shader, GL_INFO_LOG_LENGTH, byref(temp))
			# create a buffer for the log
			buffer = create_string_buffer(temp.value)
			# retrieve the log text
			glGetShaderInfoLog(shader, temp, None, buffer)
			# print the log to the console
			# print buffer.value
		else:
			# all is well, so attach the shader to the program
			glAttachShader(self.handle, shader);

	def link(self):
		# link the program
		glLinkProgram(self.handle)

		temp = c_int(0)
		# retrieve the link status
		glGetProgramiv(self.handle, GL_LINK_STATUS, byref(temp))

		# if linking failed, print the log
		if not temp:
			#	retrieve the log length
			glGetProgramiv(self.handle, GL_INFO_LOG_LENGTH, byref(temp))
			# create a buffer for the log
			buffer = create_string_buffer(temp.value)
			# retrieve the log text
			glGetProgramInfoLog(self.handle, temp, None, buffer)
			# print the log to the console
			# print buffer.value
		else:
			# all is well, so we are linked
			self.linked = True

	def bind(self):
		# bind the program
		glUseProgram(self.handle)

	def unbind(self):
		# unbind whatever program is currently bound - not necessarily this program,
		# so this should probably be a class method instead
		glUseProgram(0)

	# upload a floating point uniform
	# this program must be currently bound
	def uniformf(self, name, *vals):
		# check there are 1-4 values
		if len(vals) in range(1, 5):
			# select the correct function
			{ 1 : glUniform1f,
				2 : glUniform2f,
				3 : glUniform3f,
				4 : glUniform4f
				# retrieve the uniform location, and set
			}[len(vals)](glGetUniformLocation(self.handle, name), *vals)

	# upload an integer uniform
	# this program must be currently bound
	def uniformi(self, name, *vals):
		# check there are 1-4 values
		if len(vals) in range(1, 5):
			# select the correct function
			{ 1 : glUniform1i,
				2 : glUniform2i,
				3 : glUniform3i,
				4 : glUniform4i
				# retrieve the uniform location, and set
			}[len(vals)](glGetUniformLocation(self.handle, name), *vals)

	# upload a uniform matrix
	# works with matrices stored as lists,
	# as well as euclid matrices
	def uniform_matrixf(self, name, mat):
		# obtian the uniform location
		loc = glGetUniformLocation(self.Handle, name)
		# uplaod the 4x4 floating point matrix
		glUniformMatrix4fv(loc, 1, False, (c_float * 16)(*mat))

class MyShader(Shader):
    def __enter__(self):
        self.bind()
        return self

    def __exit__(self, *args):
        self.unbind()

class MyWindow(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(MyWindow, self).__init__(**kwargs)
        # frag = pyglet.resource.file('water.frag').read()
        self.shader = MyShader(frag=[frag])
        self.time = 0.0

    def on_draw(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.2, 0.8, 1.0, 1.0)
        self.clear()
        with self.shader:
            self.shader.uniformf('time', self.time)
            self.shader.uniformf('surface_y', float(self.height // 2))
            self.shader.uniformf('wave_height', 15.0)
            self.shader.uniformf('wave_length', 100.0)
            self.shader.uniformf('wave_speed', 0.15)
            glBegin(GL_QUADS)
            glColor4f(0.0, 0.2, 0.3, 1.0)
            glTexCoord2i(0, 0)
            glVertex2i(0, 0)
            glTexCoord2i(self.width, 0)
            glVertex2i(self.width, 0)
            glColor4f(0.0, 0.8, 1.0, 1.0)
            glTexCoord2i(self.width, self.height)
            glVertex2i(self.width, self.height)
            glTexCoord2i(0, self.height)
            glVertex2i(0, self.height)
            glEnd()

    def step(self, dt):
        self.time += dt

def main():
    config = pyglet.gl.Config(double_buffer=True, sample_buffers=1, samples=4,
                              depth_size=8)
    window = MyWindow(fullscreen=True, config=config)
    pyglet.clock.schedule_interval(window.step, 1.0 / 60.0)
    pyglet.app.run()

if __name__ == '__main__':
    main()