//#include <windows.h>
#include <stdio.h>
#include <cvGL/glew.h>
#include <cvGL/glut.h>


#include <ctype.h>
#include "opencv/cv.h"
#include "opencv/highgui.h"

GLuint texture; //the array for our texture
IplImage *imagenText1;
GLfloat angle = 0.0;

//The IplImage to OpenGl texture function
int loadTexture_Ipl(IplImage *image, GLuint *text) {

    if (image==NULL) return -1;

    glGenTextures(1, text);

    glBindTexture( GL_TEXTURE_2D, *text ); //bind the texture to it's array
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image->width, image->height,0, GL_BGR, GL_UNSIGNED_BYTE, image->imageData);
    return 0;

}

void plane (void) {
    glBindTexture( GL_TEXTURE_2D, texture ); //bind the texture
    glRotatef( angle, 1.0f, 1.0f, 1.0f );
    glBegin (GL_QUADS);
    glTexCoord2d(0.0,0.0); glVertex2d(-1.0,-1.0); //with our vertices we have to assign a texcoord
    glTexCoord2d(1.0,0.0); glVertex2d(+1.0,-1.0); //so that our texture has some points to draw to
    glTexCoord2d(1.0,1.0); glVertex2d(+1.0,+1.0);
    glTexCoord2d(0.0,1.0); glVertex2d(-1.0,+1.0);
    glEnd();

}

void display (void) {

    glClearColor (0.0,0.0,0.0,1.0);
    glClear (GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    gluLookAt (0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    glEnable( GL_TEXTURE_2D ); //enable 2D texturing
    plane();
    glutSwapBuffers();
    angle =angle+0.01;
}

void FreeTexture( GLuint texture )
{
  glDeleteTextures( 1, &texture );
}

void reshape (int w, int h) {
    glViewport (0, 0, (GLsizei)w, (GLsizei)h);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    gluPerspective (60, (GLfloat)w / (GLfloat)h, 1.0, 100.0);
    glMatrixMode (GL_MODELVIEW);
}

int main (int argc, char **argv) {
    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize (500, 500);
    glutInitWindowPosition (100, 100);
    glutCreateWindow ("A basic OpenGL Window");
    glutDisplayFunc (display);
    glutIdleFunc (display);
    glutReshapeFunc (reshape);

    imagenText1=cvLoadImage("D:/Develop/DL/projects/digital-image-processing/datas/bird.jpg");

    //The load iplimage to opengl texture
    loadTexture_Ipl( imagenText1, &texture );

    glutMainLoop ();

      //Free our texture
    FreeTexture( texture );

    return 0;
}
