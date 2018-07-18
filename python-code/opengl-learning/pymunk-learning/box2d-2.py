# -*- coding: utf-8 -*-

from __future__ import division

import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from Box2D import *

#CONTROLES

##########################################
#TECLAS DIRECCION E (W,A,S,D) -> MOVEMENTO DA BOLA
#BOTON ESQUERDO DO RATO -> CREAR RECTANGULOS
#RODA DO RATO -> ZOOM
#SUPR -> REINICIAR O XOGO
##########################################

#CONSTANTES

ANCHO_VENTANA = 600
ALTO_VENTANA = 400

LINHA_BORRADO_Y = -100

TEMPO_ESPERA_MOUSE = 10

#MEDIDA OPENGL

ANCHO_GL = 100
ALTO_GL = ANCHO_GL*ALTO_VENTANA/ANCHO_VENTANA

COLOR_FONDO = 0

#VARIABLES

pos_camara = [0,0]

descanso_mouse = 0

lista_caixas = []
lista_caixas_shape = []
lista_suelo = []

vertices_clicados = []
angulo_rectangulo_clicado = 0

modo_debuxo = "bloque"
modo_camara = "personaje"

game_over = False

#CLASES

class generador_caixas:
	def __init__(self,pos,ancho,alto,cont0,cont,densidad,friccion):
		self.pos = pos
		self.ancho = ancho
		self.alto = alto
		self.cont0 = cont0
		self.cont = cont
		self.densidad = densidad
		self.friccion = friccion

#BOX2D

FRICCION = 0.2

mundo = b2World(gravity=(0, -30))


def crear_mundo():
	global lista_suelo
	lista_suelo.append(mundo.CreateStaticBody(position=(0, 0), shapes=b2PolygonShape(box=(12,2))))
	lista_suelo.append(mundo.CreateStaticBody(position=(30, 8), angle=-0.2, shapes=b2PolygonShape(box=(10,2))))
	lista_suelo.append(mundo.CreateStaticBody(position=(60, 20), angle=0.2, shapes=b2PolygonShape(box=(10,2))))
	lista_suelo.append(mundo.CreateStaticBody(position=(90, 30), shapes=b2PolygonShape(box=(10,2))))
	lista_suelo.append(mundo.CreateStaticBody(position=(120, 25), shapes=b2PolygonShape(box=(5,2))))
	lista_suelo.append(mundo.CreateStaticBody(position=(130, 20), shapes=b2PolygonShape(box=(5,2))))
	lista_suelo.append(mundo.CreateStaticBody(position=(140, 15), shapes=b2PolygonShape(box=(5,2))))
	
crear_mundo()

lista_generador_caixas = [generador_caixas([70,120],5,2,200,0,0.05,1)]

personaje = mundo.CreateDynamicBody(position=(0,3.5), shapes=b2CircleShape(radius=1.5))

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
	global personaje
	global vertices_clicados
	global game_over
	global cont_crear_caixa
	global angulo_rectangulo_clicado
	global modo_debuxo
	global modo_camara
	
	pygame.init()
	ventana = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA), DOUBLEBUF|OPENGL)
	
	pygame.display.set_caption("Xogo de Plataformas")
	
	init_gl()
	
	fps = 60
	vel_iters = 10
	pos_iters = 10

	while True:
	
		reloj = pygame.time.Clock()
		
		mundo.Step(1 / fps, vel_iters, pos_iters)
		
		#LIMPIAR VENTANA ######
		
		limpiar_ventana()
		
		#CREAR CAIXAS
		
		for i in lista_generador_caixas:
			if not i.cont:
				lista_caixas.append(mundo.CreateDynamicBody
					(position=(i.pos[0],i.pos[1])))
				lista_caixas_shape.append(lista_caixas[-1].CreatePolygonFixture(box=(i.ancho,i.alto), density=i.densidad, friction=i.friccion))
				i.cont = i.cont0
			else:
				i.cont -= 1
		
		######################################
		#DEBUXAR #############################
		######################################
		
		glColor3f(1.0, 1.0, 1.0)
		
		for i in lista_caixas:
			debuxar_rect(i.position, list(i.fixtures[0].shape), i.angle)
		
		glColor3f(0.0, 0.5, 1.0)
		
		for i in lista_suelo:
			debuxar_rect(i.position, list(i.fixtures[0].shape), i.angle)
			
		glColor3f(0.1, 0.9, 0.1)
		
		debuxar_circulo(personaje.position, 1.5)
		
		for i in vertices_clicados:
			debuxar_punto(i)
			
		if len(vertices_clicados) >= 1:
			debuxar_rectangulo_a_pintar(vertices_clicados+[pos_mouse_gl],angulo_rectangulo_clicado)
		
		if game_over:		
			debuxar_fondo_game_over()
		
		#BORRADO DE CAIXAS
		
		for i in range(len(lista_caixas)):
			if lista_caixas[i].position[1] < LINHA_BORRADO_Y:
				lista_caixas[i].DestroyFixture(lista_caixas_shape[i])
				mundo.DestroyBody(lista_caixas[i])
				lista_caixas.remove(lista_caixas[i])
				lista_caixas_shape.remove(lista_caixas_shape[i])
				break
		
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
			
		pos_mouse_gl = [pos_mouse[0]*ANCHO_GL/ANCHO_VENTANA-ANCHO_GL/2-pos_camara[0],ALTO_GL-(pos_mouse[1]*ALTO_GL/ALTO_VENTANA)-pos_camara[1]]
		pos_mouse_gl_int = [int(pos_mouse_gl[0]),int(pos_mouse_gl[1])]
		
		if descanso_mouse == 0 and (teclas_mouse_pulsadas[0] or teclas_mouse_pulsadas[2]):
			if teclas_mouse_pulsadas[0] and len(vertices_clicados) <= 1:
				vertices_clicados.append(pos_mouse_gl_int)
				descanso_mouse = TEMPO_ESPERA_MOUSE
			if teclas_mouse_pulsadas[2]:
				vertices_clicados = []
				angulo_rectangulo_clicado = 0
		
		#TECLAS ######
		
		tecla_pulsada = pygame.key.get_pressed()
		
		if (tecla_pulsada[K_UP] or tecla_pulsada[K_w]): 
			if modo_camara == "personaje":
				if (list(personaje.linearVelocity)[1] < 0.1 and personaje.contacts and personaje.contacts[0].contact.touching
					and personaje.fixtures[0].body.contacts[0].contact.manifold.localNormal[1] >= 0.1):
					personaje.ApplyForceToCenter(b2Vec2(0,10), True)
					personaje.ApplyLinearImpulse(b2Vec2(0,35), personaje.position, False)
			else:
				pos_camara[1] -= 1
			
		
		if tecla_pulsada[K_DOWN] or tecla_pulsada[K_s]:
			if modo_camara == "personaje":
				personaje.ApplyForceToCenter(b2Vec2(0,-50), True)
			else:
				pos_camara[1] += 1
		
		if tecla_pulsada[K_LEFT] or tecla_pulsada[K_a]:
			if modo_camara == "personaje":
				if personaje.fixtures[0].body.contacts:
					if (list(personaje.linearVelocity)[0]) > -1:
						personaje.ApplyLinearImpulse(b2Vec2(-2,0), personaje.position, False)
					personaje.ApplyForceToCenter(b2Vec2(-30,0), True)
				else:
					personaje.ApplyForceToCenter(b2Vec2(-20,0), True)
					personaje.linearVelocity = b2Vec2(max(-25,list(personaje.linearVelocity)[0]),list(personaje.linearVelocity)[1])
			else:
				pos_camara[0] += 1
			
		elif tecla_pulsada[K_RIGHT] or tecla_pulsada[K_d]:
			if modo_camara == "personaje":
				if personaje.fixtures[0].body.contacts:
					if (list(personaje.linearVelocity)[0]) < 1:
						personaje.ApplyLinearImpulse(b2Vec2(2,0), personaje.position, False)
					personaje.ApplyForceToCenter(b2Vec2(30,0), True)
				else:
					personaje.ApplyForceToCenter(b2Vec2(20,0), True)
					personaje.linearVelocity = b2Vec2(min(25,list(personaje.linearVelocity)[0]),list(personaje.linearVelocity)[1])
			else:
				pos_camara[0] -= 1
				
		else:
			if personaje.fixtures[0].body.contacts:
				personaje.ApplyLinearImpulse(b2Vec2(-(list(personaje.linearVelocity)[0])/5,0), personaje.position, False)
			else:
				personaje.ApplyForceToCenter(b2Vec2(-(list(personaje.linearVelocity)[0])/20,0), True)
				
		personaje.linearVelocity = b2Vec2(max(-35,list(personaje.linearVelocity)[0]),list(personaje.linearVelocity)[1])
		personaje.linearVelocity = b2Vec2(min(35,list(personaje.linearVelocity)[0]),list(personaje.linearVelocity)[1])
		
		
		if tecla_pulsada[K_e] and len(vertices_clicados) >= 1:
			angulo_rectangulo_clicado -= 0.02
		elif tecla_pulsada[K_q] and len(vertices_clicados) >= 1:
			angulo_rectangulo_clicado += 0.02
			
			
		#personaje.linearVelocity = (20,personaje.linearVelocity[1])
		#personaje.ApplyLinearImpulse(b2Vec2(100,0), personaje.position, 0)
		
		for event in pygame.event.get():
		
			if event.type == pygame.MOUSEBUTTONDOWN:
				if event.button == 4:
					ANCHO_GL -= 5
				if event.button == 5:
					ANCHO_GL += 5
				ANCHO_GL = max(50,ANCHO_GL)
				ANCHO_GL = min(300,ANCHO_GL)
				ALTO_GL = ANCHO_GL*ALTO_VENTANA/ANCHO_VENTANA

			if event.type == pygame.KEYDOWN:
			
				if event.key == pygame.K_DELETE:
					lista_caixas = []
					lista_caixas_shape = []
					lista_suelo = []
					mundo = b2World(gravity=(0, -50))
					crear_mundo()
					personaje = mundo.CreateDynamicBody(position=(0,3.5), shapes=b2CircleShape(radius=1.5))
					pos_camara = [-personaje.position[0], -personaje.position[1]+ALTO_GL/2]
					vertices_clicados = []
					game_over = False
					
				if event.key == pygame.K_f:
					if modo_debuxo == "bloque":
						modo_debuxo = "caixa"
					else:
						modo_debuxo = "bloque"
						
				if event.key == pygame.K_c:
					if modo_camara == "personaje":
						modo_camara = "libre"
					else:
						modo_camara = "personaje"
			
				if event.key == pygame.K_SPACE:
					if len(vertices_clicados) <= 1:
						vertices_clicados = []
					elif abs(vertices_clicados[0][0] - vertices_clicados[1][0]) and abs(vertices_clicados[0][1] - vertices_clicados[1][1]):
					
						vertices_clicados = sorted(vertices_clicados, key = lambda x: x[0])
						
						pos_rectangulo_debuxado = [(vertices_clicados[0][0]+vertices_clicados[1][0])/2,(vertices_clicados[0][1]+vertices_clicados[1][1])/2]
						tamanho_rectangulo_pintado = [(vertices_clicados[1][0]-vertices_clicados[0][0])/2, 
							(sorted(vertices_clicados, key = lambda x : x[1])[1][1]-sorted(vertices_clicados, key = lambda x : x[1])[0][1])/2]
						vertices_clicados = sorted(vertices_clicados, key = lambda x: x[0])
							
						if modo_debuxo == "bloque":
							lista_suelo.append(mundo.CreateStaticBody(
								position=(pos_rectangulo_debuxado[0],pos_rectangulo_debuxado[1]), 
								angle = angulo_rectangulo_clicado,
								shapes=b2PolygonShape(box=(
											tamanho_rectangulo_pintado[0],
											tamanho_rectangulo_pintado[1])
											)))
						else:
							lista_caixas.append(mundo.CreateDynamicBody(
								position=(pos_rectangulo_debuxado[0],pos_rectangulo_debuxado[1]),
								angle = angulo_rectangulo_clicado))

							lista_caixas_shape.append(lista_caixas[-1].CreatePolygonFixture(box=(
								tamanho_rectangulo_pintado[0], tamanho_rectangulo_pintado[1]),
								density=0.2, friction=1))
								
						vertices_clicados = []
						angulo_rectangulo_clicado = 0
						
					else:
						vertices_clicados = []
						angulo_rectangulo_clicado = 0
				
			if event.type == pygame.QUIT:
				pygame.quit()
				return
                
		#ACTUALIZAR VENTANA ######
		
		pygame.display.flip()
		
		#AXUSTAR CAMARA
		
		if modo_camara == "personaje":
			if not personaje.position[1] < LINHA_BORRADO_Y:
				pos_camara = [-personaje.position[0], -personaje.position[1]+ALTO_GL/2]
			else:
				pos_camara = [pos_camara[0], -personaje.position[1]+ALTO_GL/2]
		 
			pos_camara[1] = min(pos_camara[1],-LINHA_BORRADO_Y)
			
		
		if list(personaje.position)[1] < LINHA_BORRADO_Y:
			game_over = True
		
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
	
def debuxar_circulo(pos, radio):
	glLoadIdentity()
	glEnable(GL_POLYGON_SMOOTH)
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
	glBegin(GL_POLYGON)
	for i in range(50):
		angulo = 2 * math.pi * i /50
		px = radio * math.cos(angulo)
		py = radio * math.sin(angulo)
		glVertex2f((px+pos[0]),(py+pos[1]))
	glEnd()
	
def debuxar_linea_borrado():
	glLoadIdentity()
	glColor4f(1, 0, 0, 0.8)
	glBegin(GL_LINES)
	glVertex2f(-ANCHO_GL+personaje.position[0], LINHA_BORRADO_Y)
	glVertex2f(ANCHO_GL+personaje.position[0], LINHA_BORRADO_Y)
	glEnd()
	
def debuxar_punto(pos):
	glLoadIdentity()
	glPointSize(2)
	glColor3f(0.5, 1, 1)
	glBegin(GL_POINTS)
	glVertex2f(pos[0], pos[1])
	glEnd()

def debuxar_rectangulo_a_pintar(vertices,angulo):
	ancho_cadro = vertices[1][0] - vertices[0][0]
	alto_cadro = vertices[0][1] - vertices[1][1]
	pos = [vertices[0][0]+ancho_cadro/2, vertices[1][1]+alto_cadro/2]
	glLoadIdentity()
	if modo_debuxo == "bloque":
		glColor4f(0.5, 0.5, 1, 0.2)
	else:
		glColor4f(0.8, 0.8, 0.8, 0.2)
	glTranslatef(pos[0], pos[1], 0)
	glRotatef(math.degrees(angulo),0, 0, 1)
	glRectf(vertices[0][0]-pos[0],vertices[0][1]-pos[1],vertices[1][0]-pos[0],vertices[1][1]-pos[1])
	
def debuxar_texto():
	glLoadIdentity()
	glColor3f(1, 0, 0)
	glRasterPos2f(0, 0)
	glutBitmapString(GLUT_BITMAP_HELVETICA_10, "game over")
	
def debuxar_fondo_game_over():
	glLoadIdentity()
	glColor4f(1, 0, 0, 0.2)
	glRectf(-ANCHO_GL-pos_camara[0],ALTO_GL-pos_camara[1],ANCHO_GL-pos_camara[0],LINHA_BORRADO_Y)
	
if __name__ == '__main__':
	main()