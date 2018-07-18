from __future__ import division

import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from Box2D import *

#CONTROLES

##########################################
#TECLAS DIRECCION E (W,A,S,D) -> MOVEMENTO DA CAMARA
#BOTON ESQUERDO DO RATO -> CREAR CAIXA
#BOTON DEREITO DO RATO -> CREAR BLOQUE
#RODA DO RATO -> ZOOM
#SUPR -> BORRAR CAIXAS E BLOQUES
##########################################

#CONSTANTES

ANCHO_VENTANA = 600
ALTO_VENTANA = 400

LINHA_BORRADO_Y = -100

TEMPO_ESPERA_MOUSE = 0

#MEDIDA OPENGL

ANCHO_GL = 200
ALTO_GL = ANCHO_GL*ALTO_VENTANA/ANCHO_VENTANA

COLOR_FONDO = 0

#VARIABLES

pos_camara = [0,0]

descanso_mouse = 0

lista_caixas = []
lista_caixas_shape = []
lista_suelo = []

#BOX2D

FRICCION = 0.2

mundo = b2World(gravity=(0, -30))

lista_suelo.append(mundo.CreateStaticBody(position=(0, 0), shapes=b2PolygonShape(box=(50,5))))
lista_suelo.append(mundo.CreateStaticBody(position=(-50, 15), shapes=b2PolygonShape(box=(5,20))))
lista_suelo.append(mundo.CreateStaticBody(position=(50, 15), shapes=b2PolygonShape(box=(5,20))))

#MAIN

def main():

	global ANCHO_GL
	global ALTO_GL
	global pos_camara
	global descanso_mouse
	global lista_caixas
	global lista_caixas_shape
	global lista_suelo
	global mundo
	
	pygame.init()
	ventana = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA), DOUBLEBUF|OPENGL)
	
	init_gl()
	
	fps = 60
	vel_iters = 10
	pos_iters = 10

	while True:
	
		reloj = pygame.time.Clock()
		
		mundo.Step(1 / fps, vel_iters, pos_iters)
		
		#mundo.ClearForces()
		
		#LIMPIAR VENTANA ######
		
		limpiar_ventana()
		
		#DEBUXAR ######
		
		glColor3f(1.0, 1.0, 1.0)
		
		for i in range(len(lista_caixas)):
			debuxar_rect(lista_caixas[i].position, list(lista_caixas_shape[i].shape), lista_caixas[i].angle)
		
		glColor3f(0.0, 0.5, 1.0)
		
		for i in lista_suelo:
			debuxar_rect(i.position, list(i.fixtures[0].shape))
		
		#BORRADO DE CAIXAS
		
		for i in range(len(lista_caixas)):
			if lista_caixas[i].position[1] < LINHA_BORRADO_Y:
				lista_caixas[i].DestroyFixture(lista_caixas_shape[i])
				mundo.DestroyBody(lista_caixas[i])
				lista_caixas.remove(lista_caixas[i])
				lista_caixas_shape.remove(lista_caixas_shape[i])
				break
				
		debuxar_linea_borrado()

		#############################################################
		#EVENTOS ####################################################
		#############################################################
		
		#MOUSE ######
		
		if descanso_mouse > 0:
			descanso_mouse -= 1
		
		if descanso_mouse == 0:
			pos_mouse = pygame.mouse.get_pos()
		
		if descanso_mouse == 0:
			teclas_mouse_pulsadas = pygame.mouse.get_pressed()
		
		if descanso_mouse == 0 and (teclas_mouse_pulsadas[0] or teclas_mouse_pulsadas[2]):
			if teclas_mouse_pulsadas[0]:
				lista_caixas.append(mundo.CreateDynamicBody
					(position=(pos_mouse[0]*ANCHO_GL/ANCHO_VENTANA-ANCHO_GL/2-pos_camara[0],ALTO_GL-(pos_mouse[1]*ALTO_GL/ALTO_VENTANA)-pos_camara[1])))
				lista_caixas_shape.append(lista_caixas[-1].CreatePolygonFixture(box=(1,1), density=0.5, friction=0.1))
			if teclas_mouse_pulsadas[2]:
				lista_suelo.append(mundo.CreateStaticBody
				(position=(pos_mouse[0]*ANCHO_GL/ANCHO_VENTANA-ANCHO_GL/2-pos_camara[0],ALTO_GL-(pos_mouse[1]*ALTO_GL/ALTO_VENTANA)-pos_camara[1]),
				shapes=b2PolygonShape(box=(1,1))))
			descanso_mouse = TEMPO_ESPERA_MOUSE
		
		#TECLAS ######
		
		tecla_pulsada = pygame.key.get_pressed()
		
		if tecla_pulsada[K_UP] or tecla_pulsada[K_w]:
			pos_camara[1] -= 2
		
		if tecla_pulsada[K_DOWN] or tecla_pulsada[K_s]:
			pos_camara[1] += 2
		
		if tecla_pulsada[K_LEFT] or tecla_pulsada[K_a]:
			pos_camara[0] += 2
		
		if tecla_pulsada[K_RIGHT] or tecla_pulsada[K_d]:
			pos_camara[0] -= 2
			
		if tecla_pulsada[K_SPACE]:
			for i in lista_caixas:
				i.ApplyForce(b2Vec2(0,150), i.position, 0)
		
		for event in pygame.event.get():
		
			if event.type == pygame.MOUSEBUTTONDOWN:
				if event.button == 4:
					ANCHO_GL -= 5
				if event.button == 5:
					ANCHO_GL += 5
				ANCHO_GL = max(50,ANCHO_GL)
				ANCHO_GL = min(1000,ANCHO_GL)
				ALTO_GL = ANCHO_GL*ALTO_VENTANA/ANCHO_VENTANA

			if event.type == pygame.KEYDOWN:
			
				if event.key == pygame.K_DELETE:
					lista_caixas = []
					lista_caixas_shape = []
					lista_suelo = []
					mundo = b2World(gravity=(0, -50))
					lista_suelo.append(mundo.CreateStaticBody(position=(0, 0), shapes=b2PolygonShape(box=(50,5))))
					lista_suelo.append(mundo.CreateStaticBody(position=(-50, 15), shapes=b2PolygonShape(box=(5,20))))
					lista_suelo.append(mundo.CreateStaticBody(position=(50, 15), shapes=b2PolygonShape(box=(5,20))))
					
			if event.type == pygame.QUIT:
				pygame.quit()
				return
                
		#ACTUALIZAR VENTANA ######
		
		pygame.display.set_caption("caixas: "+str(len(lista_caixas)))
		
		pygame.display.flip()
		
		reloj.tick(fps)

##############################################################
#OPENGL    ######################################################
##############################################################

def init_gl():
	glViewport(0, 0, ANCHO_VENTANA, ALTO_VENTANA)
	glClearColor(COLOR_FONDO, COLOR_FONDO, COLOR_FONDO, 1)
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
def limpiar_ventana():
	glClear(GL_COLOR_BUFFER_BIT)
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	gluOrtho2D(-ANCHO_GL/2, ANCHO_GL/2, 0, ALTO_GL)
	glTranslatef(0 + pos_camara[0],0 + pos_camara[1], 0)
	glMatrixMode(GL_MODELVIEW)

def debuxar_rect(pos, vertices, angulo=False):
	glLoadIdentity()
	glTranslatef(pos[0], pos[1], 0)
	if angulo:
		glRotatef(math.degrees(angulo), 0, 0, 1)
	glBegin(GL_QUADS)
	glVertex2f(vertices[0][0], vertices[0][1])
	glVertex2f(vertices[1][0], vertices[1][1])
	glVertex2f(vertices[2][0], vertices[2][1])
	glVertex2f(vertices[3][0], vertices[3][1])
	glEnd()
	
def debuxar_linea_borrado():
	glLoadIdentity()
	glColor4f(1, 0, 0, 0.8)
	glBegin(GL_LINES)
	glVertex2f(-100, LINHA_BORRADO_Y)
	glVertex2f(100, LINHA_BORRADO_Y)
	glEnd()
	
if __name__ == '__main__':
	main()