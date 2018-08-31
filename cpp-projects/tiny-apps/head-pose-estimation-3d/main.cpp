/***************************************************************************
 *   Copyright (C) 2007 by pedromartins   *
 *   pedromartins@isr.uc.pt   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/



//Compile with OpenGL and OpenCV libs
// g++ glAnthropometric3DModel.cpp -I /usr/local/include/opencv/ -L /usr/local/lib -lcv -lhighgui -lcvaux -lX11 -lXi -lXmu -lglut -lGL -lGLU -lm -o glAnthropometric3DModel




#include<cvGL/glew.h>
//#include <GL/gl.h>
//#include <GL/glu.h>
#include <cvGL/glut.h>
#include <stdlib.h>
#include <stdio.h>

#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>

#define cvM32f(Matrix,i,j) CV_MAT_ELEM(*(Matrix),float,(i),(j))


//Angle Conversion Macros
#define PI (3.141592653589793)

#define rad2deg(Theta) (Theta)*180/PI
#define deg2rad(Theta) (Theta)*PI/180


//Display Matrix
void cvPrintMat(CvMat*Matrix){
    int i,MatrixType;
    register int j;

    for(i=0;i<Matrix->rows;i++){		//For Each Line
        for(j=0;j<Matrix->cols;j++) printf("%g ",cvM32f(Matrix,i,j));	//For Each Columm
        printf("\n");
        }

}




float Model3D[58][3]={{-7.308957,0.913869,0.000000}, {-6.775290,-0.730814,-0.012799}, {-5.665918,-3.286078,1.022951}, {-5.011779,-4.876396,1.047961}, {-4.056931,-5.947019,1.636229}, {-1.833492,-7.056977,4.061275}, {0.000000,-7.415691,4.070434}, {1.833492,-7.056977,4.061275}, {4.056931,-5.947019,1.636229}, {5.011779,-4.876396,1.047961}, {5.665918,-3.286078,1.022951}, {6.775290,-0.730814,-0.012799}, {7.308957,0.913869,0.000000}, {5.311432,5.485328,3.987654}, {4.461908,6.189018,5.594410}, {3.550622,6.185143,5.712299}, {2.542231,5.862829,4.687939}, {1.789930,5.393625,4.413414}, {2.693583,5.018237,5.072837}, {3.530191,4.981603,4.937805}, {4.490323,5.186498,4.694397}, {-5.311432,5.485328,3.987654}, {-4.461908,6.189018,5.594410}, {-3.550622,6.185143,5.712299}, {-2.542231,5.862829,4.687939}, {-1.789930,5.393625,4.413414}, {-2.693583,5.018237,5.072837}, {-3.530191,4.981603,4.937805}, {-4.490323,5.186498,4.694397}, {1.330353,7.122144,6.903745}, {2.533424,7.878085,7.451034}, {4.861131,7.878672,6.601275}, {6.137002,7.271266,5.200823}, {6.825897,6.760612,4.402142}, {-1.330353,7.122144,6.903745}, {-2.533424,7.878085,7.451034}, {-4.861131,7.878672,6.601275}, {-6.137002,7.271266,5.200823}, {-6.825897,6.760612,4.402142}, {-2.774015,-2.080775,5.048531}, {-0.509714,-1.571179,6.566167}, {0.000000,-1.646444,6.704956}, {0.509714,-1.571179,6.566167}, {2.774015,-2.080775,5.048531}, {0.589441,-2.958597,6.109526}, {0.000000,-3.116408,6.097667}, {-0.589441,-2.958597,6.109526}, {-0.981972,4.554081,6.301271}, {-0.973987,1.916389,7.654050}, {-2.005628,1.409845,6.165652}, {-1.930245,0.424351,5.914376}, {-0.746313,0.348381,6.263227}, {0.000000,0.000000,6.763430}, {0.746313,0.348381,6.263227}, {1.930245,0.424351,5.914376}, {2.005628,1.409845,6.165652}, {0.973987,1.916389,7.654050}, {0.981972,4.554081,6.301271}};



int MeanDelaunayTriangles[95][3]={ {0,1,50}, {0,21,38}, {0,28,21}, {0,49,28}, {0,50,49}, {1,2,39}, {1,39,50}, {2,3,39}, {3,4,46}, {3,46,39}, {4,5,46}, {5,6,45}, {5,45,46}, {6,7,44}, {6,44,45}, {7,8,44}, {8,9,43}, {8,43,44}, {9,10,43}, {10,11,43}, {11,12,54}, {11,54,43}, {12,13,20}, {12,20,55}, {12,33,13}, {12,55,54}, {13,14,20}, {13,33,14}, {14,15,19}, {14,19,20}, {14,31,15}, {14,32,31}, {14,33,32}, {15,16,19}, {15,30,16}, {15,31,30}, {16,17,18}, {16,18,19}, {16,29,17}, {16,30,29}, {17,29,57}, {17,57,18}, {18,56,19}, {18,57,56}, {19,55,20}, {19,56,55}, {21,22,38}, {21,28,22}, {22,23,36}, {22,27,23}, {22,28,27}, {22,36,37}, {22,37,38}, {23,24,35}, {23,26,24}, {23,27,26}, {23,35,36},
{24,25,34}, {24,26,25}, {24,34,35}, {25,26,48}, {25,47,34}, {25,48,47}, {26,27,49}, {26,49,48}, {27,28,49}, {29,30,35}, {29,34,57}, {29,35,34}, {30,31,36}, {30,36,35}, {34,47,57}, {39,40,50}, {39,46,40}, {40,41,52}, {40,46,41}, {40,51,50}, {40,52,51}, {41,42,52}, {41,44,42},
{41,45,44}, {41,46,45}, {42,43,53}, {42,44,43}, {42,53,52}, {43,54,53}, {47,48,57}, {48,49,51}, {48,51,52}, {48,52,56}, {48,56,57}, {49,50,51}, {52,53,56}, {53,54,55}, {53,55,56}};









void Anthropometric3DModel(int LineWidth){
    int k;

    glColor3f(0.0, 0.0, 0.0);

    //glLineWidth(2.0);
    glLineWidth(LineWidth);

glPushMatrix();

glBegin(GL_LINES);
    for(k=0;k<95;k++){

    glVertex3f(Model3D[MeanDelaunayTriangles[k][0]][0],Model3D[MeanDelaunayTriangles[k][0]][1],Model3D[MeanDelaunayTriangles[k][0]][2]);
    glVertex3f(Model3D[MeanDelaunayTriangles[k][1]][0],Model3D[MeanDelaunayTriangles[k][1]][1],Model3D[MeanDelaunayTriangles[k][1]][2]);

    glVertex3f(Model3D[MeanDelaunayTriangles[k][1]][0],Model3D[MeanDelaunayTriangles[k][1]][1],Model3D[MeanDelaunayTriangles[k][1]][2]);
    glVertex3f(Model3D[MeanDelaunayTriangles[k][2]][0],Model3D[MeanDelaunayTriangles[k][2]][1],Model3D[MeanDelaunayTriangles[k][2]][2]);

    glVertex3f(Model3D[MeanDelaunayTriangles[k][2]][0],Model3D[MeanDelaunayTriangles[k][2]][1],Model3D[MeanDelaunayTriangles[k][2]][2]);
    glVertex3f(Model3D[MeanDelaunayTriangles[k][0]][0],Model3D[MeanDelaunayTriangles[k][0]][1],Model3D[MeanDelaunayTriangles[k][0]][2]);

    }
glEnd();



    glColor3f(1.0, 0.0, 0.0);
    glPointSize(LineWidth+2);

    glBegin(GL_POINTS);
        for(k=0;k<58;k++) glVertex3f(Model3D[k][0],Model3D[k][1],Model3D[k][2]);
    glEnd();

glPopMatrix();

}





GLUquadricObj*Quadric=gluNewQuadric();

void DrawAxes(float Size,float Radius){


glPushMatrix();


    //Draw z axis
    glColor3f(0.0, 0.0, 1.0);
    glPushMatrix();
    gluCylinder(Quadric,Radius,Radius,Size,40,40);
    glTranslatef(0,0,Size);
    gluCylinder(Quadric,Radius*2,0,1,40,40);
    glPopMatrix();


    //Draw x axis
    glColor3f(1.0, 0.0, 0.0);
    glPushMatrix();
    glRotatef(90,0,1,0);	//rotate over y 90ยบ
    gluCylinder(Quadric,Radius,Radius,Size,40,40);
    glTranslatef(0,0,Size);
    gluCylinder(Quadric,Radius*2,0,1,40,40);
    glPopMatrix();

    //Draw y axis
    glColor3f(0.0, 1.0, 0.0);
    glPushMatrix();
    glRotatef(-90,1,0,0);	//rotate over x -90ยบ
    gluCylinder(Quadric,Radius,Radius,Size,40,40);
    glTranslatef(0,0,Size);
    gluCylinder(Quadric,Radius*2,0,1,40,40);
    glPopMatrix();



glPopMatrix();

}






GLfloat WorldTx=0,WorldTy=0,WorldTz=0,WorldRoll=0,WorldPitch=0,WorldYaw=0;

GLfloat FaceTx=0,FaceTy=0,FaceTz=0,FaceRoll=0,FacePitch=0,FaceYaw=0;

IplImage*Image=NULL;






//OpenGL Display Routine
void Display(void){
    int k;

    glClear(GL_COLOR_BUFFER_BIT);




glPushMatrix();

    glTranslatef(FaceTx,FaceTy,FaceTz);

    glRotatef(FaceRoll,0,0,1);
    glRotatef(FacePitch,0,1,0);
    glRotatef(FaceYaw,1,0,0);


    Anthropometric3DModel(3);

    DrawAxes(5,0.1);

glPopMatrix();




    glutSwapBuffers();


    //Read From FrameBuffer
    glReadPixels(0,0,Image->width,Image->height,GL_BGR,GL_UNSIGNED_BYTE,Image->imageData);

}






//OpenGL Idle Function
void Idle(void){
    //printf("glIdle\n");
    glutPostRedisplay();
}

void Visible(int vis){
    if (vis == GLUT_VISIBLE)
        glutIdleFunc(Idle);
    else
        glutIdleFunc(NULL);
}



void KeyboardHandler(unsigned char Key, int x, int y){

    switch(Key){

    case 27: 	cvReleaseImage(&Image);
            cvDestroyWindow("Image");
            exit(0); break;


    case 'r': WorldRoll+=5; break;
    case 'R': WorldRoll-=5; break;

    case 'p': WorldPitch+=5; break;
    case 'P': WorldPitch-=5; break;

    case 'y': WorldYaw+=5; break;
    case 'Y': WorldYaw-=5; break;





    case '7': FaceRoll+=2; break;
    case '1': FaceRoll-=2; break;

    case '9': FacePitch+=2; break;
    case '3': FacePitch-=2; break;

    case '/': FaceYaw+=2; break;
    case '*': FaceYaw-=2; break;

    case '6': FaceTx+=0.5; break;
    case '4': FaceTx-=0.5; break;

    case '8': FaceTy+=0.5; break;
    case '2': FaceTy-=0.5; break;

    case '+': FaceTz+=0.5; break;
    case '-': FaceTz-=0.5; break;




    case 's':	cvFlip(Image,Image);
            cvShowImage("Image",Image);
            cvWaitKey(50);
            char Buffer[50];
            static int FrameIndex=0;
            sprintf(Buffer,"Frame-%03d.jpg",FrameIndex++);
            cvSaveImage(Buffer,Image);
            break;



    }




    glRotatef(WorldRoll,0,0,1);
    glRotatef(WorldPitch,0,1,0);
    glRotatef(WorldYaw,1,0,0);


    WorldRoll=0;
    WorldPitch=0;
    WorldYaw=0;

    glutPostRedisplay();
}



void SpecialKeyboardHandler(int Key, int x, int y){
    switch(Key){

    case GLUT_KEY_RIGHT: WorldTx+=0.5; break;
    case GLUT_KEY_LEFT: WorldTx-=0.5; break;
    case GLUT_KEY_UP: WorldTy+=0.5; break;
    case GLUT_KEY_DOWN: WorldTy-=0.5; break;
    case GLUT_KEY_PAGE_UP: WorldTz+=0.5; break;
    case GLUT_KEY_PAGE_DOWN: WorldTz-=0.5; break;
    }

    glTranslatef(WorldTx,WorldTy,WorldTz);

    WorldTx=0;
    WorldTy=0;
    WorldTz=0;

    glutPostRedisplay();
  }





//Reshape Window Handler
void ReshapeWindow(GLsizei w,GLsizei h){
    glViewport( 0, 0, w, h);

    glMatrixMode (GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(60.0, (GLfloat) w/(GLfloat) h, 1.0, 200.0);

    //Using Orthographic Projection
    //glOrtho (0, w, 0, h, -5.0, 5.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, -25);


    //glGetDoublev(GL_MODELVIEW_MATRIX, R);

}








//Set Texture Mapping Parameters
void Init(void){

    //Clear The Background Color
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);


    glShadeModel(GL_SMOOTH);


    Image=cvCreateImage(cvSize(800,600),IPL_DEPTH_8U,3);
    cvSetZero(Image);



    cvNamedWindow("Image",CV_WINDOW_AUTOSIZE);

}










int main(int argc, char** argv){

    glutInit(&argc,argv);

    //Init OpenGL With Double Buffer in RGB Mode
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

    glutInitWindowSize(800,600);

    glutCreateWindow("3D Model");

    //Set Display Handler
    glutDisplayFunc(Display);

    //Set Keyboard Handler
    glutKeyboardFunc(KeyboardHandler);
    glutSpecialFunc(SpecialKeyboardHandler);

    glutReshapeFunc(ReshapeWindow);
    glutVisibilityFunc(Visible);

    Init();


    //OpenGL Main Loop
    glutMainLoop();

    return 0;
}










