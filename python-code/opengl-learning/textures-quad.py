# -*- coding: utf-8 -*-

import sys
from PyQt5 import QtWidgets, QtGui
import numpy

class MyOpenGLWindow( QtGui.QOpenGLWindow ):
    def __init__( self ):
        super().__init__()
        self.profile = QtGui.QOpenGLVersionProfile()
        self.profile.setVersion( 2, 1 )
        self.image = QtGui.QImage( "resources/images/city.jpg" ).mirrored()
        self.image_map = QtGui.QImage('resources/images/1977map.png').mirrored()
    def initializeGL( self ):
        self.gl = self.context().versionFunctions( self.profile )
        self.vao_offscreen = QtGui.QOpenGLVertexArrayObject( self )
        self.vao_offscreen.create()
        self.vao = QtGui.QOpenGLVertexArrayObject( self )
        self.vao.create()
        self.texture = QtGui.QOpenGLTexture( self.image )
        self.texture_map = QtGui.QOpenGLTexture(self.image_map)


        self.program = QtGui.QOpenGLShaderProgram( self )
        self.program.addShaderFromSourceFile( QtGui.QOpenGLShader.Vertex, 'resources/shaders/simple_texture.vs' )
        self.program.addShaderFromSourceFile( QtGui.QOpenGLShader.Fragment, 'resources/shaders/instagram/1997.fs' )
        self.program.link()

        self.program.bind()
        self.matrix = QtGui.QMatrix4x4()
        self.matrix.ortho( 0, 1, 0, 1, 0, 1 )
        self.program.setUniformValue( "mvp", self.matrix )
        self.program.release()

        # screen render
        self.vao.bind()
        self.vertices = [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0 ]
        self.vbo_vertices = self.setVertexBuffer( self.vertices, 3, self.program, "position" )
        self.tex = [ 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0 ]
        self.vbo_tex = self.setVertexBuffer( self.tex, 2, self.program, "texCoord" )
        self.vao.release()

    def paintGL( self ):
        self.gl.glViewport( 0, 0, self.image.width(), self.image.height() )

        self.program.bind()

        self.texture.bind()
        self.texture_map.bind()
        self.vao.bind()
        self.gl.glDrawArrays( self.gl.GL_TRIANGLE_STRIP, 0, 4 )
        self.vao.release()
        self.texture.release()
        self.texture_map.release()
        self.program.release()

    def setVertexBuffer( self, data_array, dim_vertex, program, shader_str ):
        vbo = QtGui.QOpenGLBuffer( QtGui.QOpenGLBuffer.VertexBuffer )
        vbo.create()
        vbo.bind()

        vertices = numpy.array( data_array, numpy.float32 )
        vbo.allocate( vertices, vertices.shape[0] * vertices.itemsize )

        attr_loc = program.attributeLocation( shader_str )
        program.enableAttributeArray( attr_loc )
        program.setAttributeBuffer( attr_loc, self.gl.GL_FLOAT, 0, dim_vertex )
        vbo.release()

        return vbo

if __name__ == '__main__':
    app = QtWidgets.QApplication( sys.argv )

    window = MyOpenGLWindow()
    width,height = window.image.width(),window.image.height()
    window.resize( width, height )
    window.show()

    app.exec_()
