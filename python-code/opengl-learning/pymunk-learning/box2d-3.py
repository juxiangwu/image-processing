import math
import random

import pygame
from pygame.locals import *

import Box2D
from Box2D.b2 import * 
from Box2D import *

#from framework import *
#from pygame_framework import *
import time

import copy
# print pygame.__file__
pygame.init()

groundPieceWidth = 1.5
groundPieceHeight = 0.15

chassisMaxAxis = 1.1
chassisMinAxis = 0.1
chassisMinDensity = 50
chassisMaxDensity = 100

wheelMaxRadius = 0.5
wheelMinRadius = 0.2
wheelMaxDensity = 30
wheelMinDensity = 10
motorSpeed = 25

gravity = b2Vec2(0.0, -9.81)
doSleep = True;
#print help(world)
start_position = b2Vec2(1,2)

max_health = 100

import random


class car_info:
	def __init__(self):
		self.wheel_count = 2 #default
		self.wheel_radius = [0]*self.wheel_count
		self.wheel_density = [0]*self.wheel_count
		self.wheel_vertex = [0]*self.wheel_count
		self.chassis_density = 1 #default
		self.vertex_list = [0]*8

	def get_wheel_count(self):
		return self.wheel_count
	def get_wheel_radius(self):
		return self.wheel_radius
	def get_wheel_density(self):
		return self.wheel_density
	def get_wheel_vertex(self):
		return self.wheel_vertex
	def get_chassis_density(self):
		return self.chassis_density
	def get_vertex_list(self):
		return self.vertex_list

	def set_wheel_count(self,val):
		self.wheel_count = val
	def set_wheel_radius(self,val_list):
		self.wheel_radius = val_list
	def set_wheel_density(self,val_list):
		self.wheel_density = val_list
	def set_wheel_vertex(self,val_list):
		self.wheel_vertex = val_list
	def set_chassis_density(self,val):
		self.chassis_density = val
	def set_vertex_list(self,val_list):
		self.vertex_list = val_list

"""
def make_acar(car_data_list): #car_data is the opt from random_car
	car_array = []
	for car in car_data_list:
		car_array.append()
def make_and_draw_car(n = 2):
	carGeneration = []
	for i in range(n):
		carGeneration.append(make_random_car())
"""


class car:
	def __init__(self,world,random = True,car_def = None):
		global motorSpeed,gravity,groundPieceWidth,groundPieceHeight ,chassisMaxAxis,chassisMinAxis,chassisMinDensit,chassisMaxDensity,wheelMaxRadius,wheelMinRadius,wheelMaxDensity,wheelMinDensit
		self.world = world
		if random:
			self.car_def = self.make_random_car()
		else:
			self.car_def = car_def


		self.alive = True;
		self.velocityIndex = 0;
		self.chassis = self.create_chassis(self.car_def.vertex_list, self.car_def.chassis_density)
		self.wheels = []
		for i in range(self.car_def.wheel_count):
			self.wheels.append(self.create_wheel(self.car_def.wheel_radius[i], self.car_def.wheel_density[i]))
		#carmass = 2+1 #(2 for wheels and 1 for body)
		carmass = self.chassis.mass
		for i in range(self.car_def.wheel_count):
			carmass += self.wheels[i].mass

		carmass = 2+1
		#better: theres a getMassData method for b2Body - check that!
		self.torque = []
		for i in range(self.car_def.wheel_count):
			self.torque.append(carmass * -gravity.y / self.car_def.wheel_radius[i])



		self.joint_def = b2RevoluteJointDef()


		for i in range(self.car_def.wheel_count):
			randvertex = self.chassis.vertex_list[self.car_def.wheel_vertex[i]]
			self.joint_def.localAnchorA.Set(randvertex.x, randvertex.y)
			self.joint_def.localAnchorB.Set(0, 0)
			self.joint_def.maxMotorTorque = self.torque[i]
			self.joint_def.motorSpeed = -motorSpeed
			self.joint_def.enableMotor = True
			self.joint_def.collideConnected = False
			self.joint_def.bodyA = self.chassis
			self.joint_def.bodyB = self.wheels[i]
			joint = self.world.CreateJoint(self.joint_def)

		#print "->",self.chassis.fixtures[0].type

	def make_random_car(self):
		global motorSpeed,gravity,groundPieceWidth,groundPieceHeight ,chassisMaxAxis,chassisMinAxis,chassisMinDensit,chassisMaxDensity,wheelMaxRadius,wheelMinRadius,wheelMaxDensity,wheelMinDensit
		
		random_car = car_info()

		wheel_radius_values = []
		wheel_density_values = []
		vertex_list = []
		wheel_vertex_values = []

		for i in range(random_car.get_wheel_count()):
			wheel_radius_values.append(random.random()*wheelMaxRadius+wheelMinRadius)
			wheel_density_values.append(random.random()*wheelMaxDensity+wheelMinDensity)


		vertex_list.append(b2Vec2(random.random()*chassisMaxAxis + chassisMinAxis,0))
		vertex_list.append(b2Vec2(random.random()*chassisMaxAxis + chassisMinAxis,random.random()*chassisMaxAxis + chassisMinAxis))
		vertex_list.append(b2Vec2(0,random.random()*chassisMaxAxis + chassisMinAxis))
		vertex_list.append(b2Vec2(-random.random()*chassisMaxAxis - chassisMinAxis,random.random()*chassisMaxAxis + chassisMinAxis))
		vertex_list.append(b2Vec2(-random.random()*chassisMaxAxis - chassisMinAxis,0))
		vertex_list.append(b2Vec2(-random.random()*chassisMaxAxis - chassisMinAxis,-random.random()*chassisMaxAxis - chassisMinAxis))
		vertex_list.append(b2Vec2(0,-random.random()*chassisMaxAxis - chassisMinAxis))
		vertex_list.append(b2Vec2(random.random()*chassisMaxAxis + chassisMinAxis,-random.random()*chassisMaxAxis - chassisMinAxis))

		index_left = [i for i in range(8)]
		#print index_left
		for i in range(random_car.get_wheel_count()):
			index_of_next = int(random.random() * (len(index_left)-1))
			#print index_of_next
			wheel_vertex_values.append(index_left[index_of_next])
			#remove the last used index from index_left
			index_left = index_left[:index_of_next] + index_left[index_of_next+1:]

		#now, setting all values (these are all the attibutes required to completely describe a car)
		random_car.set_vertex_list(vertex_list)
		random_car.set_wheel_radius(wheel_radius_values)
		random_car.set_wheel_density(wheel_density_values)
		random_car.set_wheel_vertex(wheel_vertex_values)
		random_car.set_chassis_density(random.random()*chassisMaxDensity+chassisMinDensity)

		return random_car
	

	def create_wheel(self,radius,density):
		body_def = bodyDef()
		body_def.type = b2_dynamicBody
		body_def.position.Set(start_position.x,start_position.y)
		body = self.world.CreateBody(body_def)
		fix_def = b2FixtureDef()
		fix_def.shape = b2CircleShape(radius = radius)
		fix_def.density = density
		fix_def.friction = 1
		fix_def.restitution = 0.2
		fix_def.filter.groupIndex = -1
		body.CreateFixture(fix_def)

		#body_def.type = b2_dynamicBody
		#body_def.position.Set(body.worldCenter.x,body.worldCenter.y)
		#line = self.world.CreateBody(body_def,angle=15)
		#box = body.CreatePolygonFixture(box=(0.01,radius/2),density = 0.00001,friction = 0)

		return body


	def create_chassis_part(self,body,vertex1,vertex2,density):
		vertex_list = []
		vertex_list.append(vertex1)
		vertex_list.append(vertex2)
		vertex_list.append(b2Vec2(0,0))
		fix_def = b2FixtureDef()
		fix_def.shape = b2PolygonShape()
		fix_def.density = density
		fix_def.friction = 10
		fix_def.restitution = 0.0
		fix_def.filter.groupIndex = -1
		#fix_def.shape.SetAsArray(vertex_list,3)
		#print "length of vertex in chassis:",len(vertex_list)
		#print vertex_list	
		fix_def.shape = b2PolygonShape(vertices=vertex_list)
		body.CreateFixture(fix_def)

	def create_chassis(self,vertex_list,density):
		body_def = b2BodyDef()
		body_def.type = b2_dynamicBody;
		body_def.position.Set(start_position.x,start_position.y)	#start position of the car
		body = self.world.CreateBody(body_def)	
		for i in range(len(vertex_list)):
			self.create_chassis_part(body, vertex_list[i],vertex_list[(i+1)%8], density)	
		body.vertex_list = vertex_list	
		return body

	def get_car_chassis(self):
		return self.chassis
	def get_car_wheels(self):
		return self.wheels

"""
	def draw_stuff(self):
		PPM=30.0 # pixels per meter
		TARGET_FPS=60
		TIME_STEP=1.0/TARGET_FPS
		SCREEN_WIDTH, SCREEN_HEIGHT=640,480
		running=True
		screen=pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT), 0, 32)
		pygame.display.set_caption('Simple pygame example')
		clock=pygame.time.Clock()
		def my_draw_circle(circle, body, fixture):
		    colors = {staticBody  : (255,255,255,255),dynamicBody : (127,127,127,255)}
		    position=body.transform*circle.pos*PPM
		    position=(position[0], SCREEN_HEIGHT-position[1])
		    pygame.draw.circle(screen, colors[body.type], [int(x) for x in position], int(circle.radius*PPM))
		b2CircleShape.draw=my_draw_circle
		while running:
		    # Check the event queue
		    for event in pygame.event.get():
		        if event.type==QUIT or (event.type==KEYDOWN and event.key==K_ESCAPE):
		            # The user closed the window or pressed escape
		            running=False
		    screen.fill((0,0,0,0))
		    # Draw the world
		    for body in self.wheels: # or: world.bodies
		    	print body.type
		        # The body gives us the position and angle of its shapes
		        for fixture in body.fixtures:
		        	fixture.shape.draw(body, fixture)
		    world.Step(TIME_STEP, 10, 10)
		    # Flip the screen and try to keep at the target FPS
		    pygame.display.flip()
		    clock.tick(TARGET_FPS)
"""
"""
def draw_any(world):
	#body aray contains the list of body objects to be draw
	#type = 1 means polygon, type = 2 means circle
	x_offset = 0
	y_offset = 0
	offset_value = 5
	PPM=30.0 # pixels per meter
	TARGET_FPS=60
	TIME_STEP=1.0/TARGET_FPS
	SCREEN_WIDTH, SCREEN_HEIGHT=640,480
	running=True
	screen=pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT), 0, 32)
	pygame.display.set_caption('Cars learning to drive')
	clock=pygame.time.Clock()
	colors = {staticBody  : (136,150,200,255),dynamicBody : (127,127,127,255)}
	def my_draw_circle(circle, body, fixture):
		#print "drawing circle"
		position=body.transform*circle.pos*PPM
		position=(position[0]+x_offset, SCREEN_HEIGHT-position[1]+y_offset)
		pygame.draw.circle(screen, colors[body.type], [int(x) for x in position], int(circle.radius*PPM))
		#pygame.draw.aaline(screen, (255,0,0), [int(x) for x in position], (center[0] - radius*axis[0], center[1] + radius*axis[1])) 
	b2CircleShape.draw=my_draw_circle
	def my_draw_polygon(polygon, body, fixture):
		#print "drawing poly"
		vertices=[(body.transform*v)*PPM for v in polygon.vertices]
		vertices=[(v[0]+x_offset, SCREEN_HEIGHT-v[1]+y_offset) for v in vertices]
		#print vertices
		#print body.type
		pygame.draw.polygon(screen, colors[body.type], vertices)
	polygonShape.draw=my_draw_polygon
	while running:
	    # Check the event queue
	    for event in pygame.event.get():
	        if event.type==QUIT or (event.type==KEYDOWN and event.key==K_ESCAPE):
	            # The user closed the window or pressed escape
	            running=False
	    pressed = pygame.key.get_pressed()
	    if pressed[pygame.K_w]:
	        y_offset +=offset_value
	    if pressed[pygame.K_s]:
	        y_offset -=offset_value
	    if pressed[pygame.K_a]:
	        x_offset +=offset_value
	    if pressed[pygame.K_d]:
	        x_offset -=offset_value
	    screen.fill((100,25,10,100))
	    # Draw the world
	    #print "world bodies:",len(body_array)
	    for body in world.bodies:
	        for fixture in body.fixtures:
	            #print "drawing.."
	            fixture.shape.draw(body, fixture)
	    # Make Box2D simulate the physics of our world for one step.
	    world.Step(TIME_STEP, 10, 10)
	    # Flip the screen and try to keep at the target FPS
	    pygame.display.flip()
	    clock.tick(TARGET_FPS)
"""
class terrain:
	def __init__(self,world):
		self.world = world

	def create_floor(self):
		maxFloorTiles = 200
		last_tile = None
		tile_position = b2Vec2(-1,0)
		floor_tiles = []
		random.seed(random.randint(1,39478534))
		for k in range(maxFloorTiles):
			last_tile = self.create_floor_tile(tile_position, (random.random()*3 - 1.5) * 1.2*k/maxFloorTiles)
			floor_tiles.append(last_tile)
			last_fixture = last_tile.fixtures
			#below is the fix for jagged edges: the vertex order was messed up, so sometimes the left bottom corner
			#would be connected to the top right corner of the previous tile
			if last_fixture[0].shape.vertices[3]==b2Vec2(0,0):
				last_world_coords = last_tile.GetWorldPoint(last_fixture[0].shape.vertices[0])
			else:
				last_world_coords = last_tile.GetWorldPoint(last_fixture[0].shape.vertices[3])
			tile_position = last_world_coords
			#print "lasttile position:",last_world_coords
		#print len(floor_tiles)
		#floor_tiles = []
		#floor_tiles.append(create_floor_tile(b2Vec2(50,50), (random.random()*3 - 1.5) * 1.2*3/maxFloorTiles))
		return floor_tiles

	def create_floor_tile(self,position, angle):
		#print "creating next tile at position: ",position
		global motorSpeed,gravity,groundPieceWidth,groundPieceHeight ,chassisMaxAxis,chassisMinAxis,chassisMinDensit,chassisMaxDensity,wheelMaxRadius,wheelMinRadius,wheelMaxDensity,wheelMinDensit
		body_def = b2BodyDef()
		#body_def.position.Set(position.x, position.y)
		body_def.position = position
		body = self.world.CreateBody(body_def)
		fix_def = b2FixtureDef()
		fix_def.shape = b2PolygonShape()
		fix_def.friction = 0.5
		coords = []
		coords.append(b2Vec2(0,0))
		coords.append(b2Vec2(0,groundPieceHeight))
		coords.append(b2Vec2(groundPieceWidth,groundPieceHeight))
		coords.append(b2Vec2(groundPieceWidth,0))
		newcoords = self.rotate_floor_tile(coords, angle)
		#print "length of polygon in tile:",len(newcoords)
		#newcoords[3].y +=0.15
		fix_def.shape = b2PolygonShape(vertices=newcoords) #setAsArray alt
		
		#print newcoords
		body.CreateFixture(fix_def)
		#print "newtile position: ",body.GetWorldPoint(body.fixtures[0].shape.vertices[0])
		#print "length of a single body's fixure list:",len(body.fixtures) # 1
		#print "all fixures:",body.fixtures #datas in fictures i.e ficture properties
		#print "shape of fixure:",body.fixtures[0].shape
		return body

	def rotate_floor_tile(self,coords, angle):
		newcoords = []
		for k in range(len(coords)):
			nc = b2Vec2(0,0)
			nc.x = math.cos(angle)*(coords[k].x) - math.sin(angle)*(coords[k].y)
			nc.y = math.sin(angle)*(coords[k].x) + math.cos(angle)*(coords[k].y)
			newcoords.append(nc)
		return newcoords



#print Box2D.RAND_LIMIT
#world = world(gravity, doSleep)
#car_def = make_random_car()
#c = car(car_def,world)
#c.draw_stuff()

#draw_any(world,create_floor(world))

#create_wheel(5,3)

#create_chassis(car_def.vertex_list, car_def.chassis_density)
class car_data:
	global start_position,max_health
	def __init__(self,chassis,wheels,car_def,xy_pos=[0,0],linear_vel=0):
		self.xy_pos = xy_pos #[x,y]
		self.linear_vel = linear_vel
		self.health = max_health
		self.isDead = False
		self.chassis = chassis
		self.wheels = wheels
		self.max_dist = 0
		self.car_def = car_def
	def kill_it(self):
		self.health = 0
		self.isDead = True
	def getHealth(self):
		return self.health
	def isDead(self):
		return self.isDead
	def dcr_health(self):
		self.health -= 2
	def get_vel(self):
		return self.linear_vel
	def get_pos_x(self):
		return self.xy_pos[0]
	def get_pos(self):
		return self.xy_pos
	def set_pos_and_vel(self,pos,vel):
		if not self.isDead:
			self.xy_pos = pos
			self.linear_vel = vel
			self.update_health()
			self.update_max_dist()
	def update_health(self):
		if self.linear_vel < 0.0001:
			self.dcr_health()
			if self.health <= 0:
				self.kill_it()
	def print_info(self):
		if not self.isDead:
			print ("Velocity:",self.linear_vel," Position:",self.xy_pos," Health:",self.health)
		else:
			print ("Dead")
	def update_max_dist(self):
		self.max_dist = self.xy_pos[0]-start_position.x
import __builtin__
#class do_stuff(Framework): #uncomment for framework stuff
class do_stuff():
	global motorSpeed,gravity,groundPieceWidth,groundPieceHeight ,chassisMaxAxis,chassisMinAxis,chassisMinDensit,chassisMaxDensity,wheelMaxRadius,wheelMinRadius,wheelMaxDensity,wheelMinDensit
	
	def __init__(self):
		self.world = b2World(gravity=(0,-9.81), doSleep=True)
#		super(do_stuff,self).__init__() #uncoment for framework stuff

		#c = car(self.world)
		#self.wheels = c.get_car_wheels()
		#self.chassis = c.get_car_chassis()
		#self.start_update = False
		#self.world = b2World(gravity,doSleep)
		self.population_size = 20
		
		self.killed = 0

		t = terrain(self.world)
		self.terrain = t.create_floor()
		
		self.population = [] #array of list of [chassis,wheels]
		self.population_data = [] #array of car_data objetcs
	#	self.population.append([self.wheels,self.chassis])
		self.create_generation_1()
		self.leader_coors = [0,0]
		self.leader = self.population[0][0] #chassis of 1st car


		self.draw_any()

		
		

		#self.update_car_data()
	def draw_any(self):
		#body aray contains the list of body objects to be draw
		#type = 1 means polygon, type = 2 means circle

		x_offset = 0
		y_offset = 0
		prev_y = 0
		offset_value = 5


		PPM=30.0 # pixels per meter
		TARGET_FPS=60
		TIME_STEP=1.0/TARGET_FPS
		SCREEN_WIDTH, SCREEN_HEIGHT=640,480
		running=True
		screen=pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT), 0, 32)
		pygame.display.set_caption('Cars learning to drive')
		clock=pygame.time.Clock()

		colors = {staticBody  : (136,150,200,255),dynamicBody : (127,127,127,255)}
		leader_coors = self.leader_coors

		def my_draw_circle(circle, body, fixture):
			#print "drawing circle"
			#help(body)
			global wheelMinDensity
			position=body.transform*circle.pos*PPM

			y_offset = ((self.leader.worldCenter.y)*70)
			if y_offset < -300:
				y_offset = -300
			if y_offset > 300:
				y_offset = 300

			position=(position[0]-self.leader.worldCenter.x*30+350, SCREEN_HEIGHT-position[1]+y_offset*0.5-200)
			
			#higher the density darker the wheel
			#c = round(255 - (255 * (body.fixtures[0].density - wheelMinDensity)) / wheelMaxDensity)
			#pygame.draw.circle(screen, (c,c,c), [int(x) for x in position], int(circle.radius*PPM),1)

			center = [int(x) for x in position]   #just (x,y) coor
												  #uncomment above if you just want to draw on the screen and without using blit (cant use alpha color value)

			center_s = [int(circle.radius*PPM),int(circle.radius*PPM)] #this is for drawing on the new surface we create below
																	  #0,0 is top left corner, so to draw a circle on the top
																	  #left corner, we set center as radius,radius

			s = pygame.Surface((50,50)) #create a surface just enough for the wheel radius , too big will cause the sim. to lag
			s.set_alpha(100)		#transparancy value
            s.fill((255,255,255))
   			s.set_colorkey((255,255,255))	#comment this to see how screen blit works

			pygame.draw.circle(s, (38, 192, 90), center_s, int(circle.radius*PPM),0)	#draw a circle on the new screen we created
			
			#pygame.draw.circle(screen, (150,150,150), center, int(circle.radius*PPM),0)	#uncomment to draw on normal screen (no alpha values)

			t = body.transform
			axis = b2Mul(t.q, b2Vec2(10.0, 25.0))
	
			pygame.draw.aaline(s, (255,0,0), center_s, (center_s[0] -circle.radius*axis[0], center_s[1] +circle.radius*axis[1]) )

   			screen.blit(s, (position[0]-int(circle.radius*PPM),position[1]-int(circle.radius*PPM)))
   			
   			#myfont = pygame.font.SysFont("impact", 10)
   			#screen.blit(myfont.render("P", True, (255,255,0)), (position[0],SCREEN_HEIGHT-position[1]))

		b2CircleShape.draw=my_draw_circle


		def my_draw_polygon(polygon, body, fixture):
			#print "drawing poly"
			#print self.leader.worldCenter.x,self.leader.worldCenter.y
			y_offset = ((self.leader.worldCenter.y)*70)
			if y_offset < -300:
				y_offset = -300
			if y_offset > 300:
				y_offset = 300
			#print y_offset
			vertices=[(body.transform*v)*PPM for v in polygon.vertices]
			vertices=[(v[0]-self.leader.worldCenter.x*30+350, SCREEN_HEIGHT-v[1]+y_offset*0.5-200) for v in vertices]
			pygame.draw.polygon(screen, colors[body.type], vertices)


		polygonShape.draw=my_draw_polygon

		while running:
			self.update_car_data()
			self.update_leader()
			if self.killed == self.population_size:
				self.next_generation()
			# Check the event queue
			for event in pygame.event.get():
				if event.type==QUIT or (event.type==KEYDOWN and event.key==K_ESCAPE):
					# The user closed the window or pressed escape
					running=False

			#screen.fill((100,25,10,100))
			screen.fill((90,23,100,100))
			#229,153,153,255
			
		    # Draw the world
			for body in self.world.bodies:
				for fixture in body.fixtures:
		            #print "drawing.."
					fixture.shape.draw(body, fixture)

		    # Make Box2D simulate the physics of our world for one step.
			self.world.Step(TIME_STEP, 10, 10)

		    # Flip the screen and try to keep at the target FPS
			pygame.display.flip()
			clock.tick(TARGET_FPS)

	def update_leader(self):
		sorted_data = sorted(self.population_data,key = lambda x:x.max_dist)
		for data in sorted_data:
			if not data.isDead:
				self.leader = data.chassis
	def Step(self, settings):
		super(do_stuff, self).Step(settings)
		#print "--"*20
		#for cars in self.population_data:
			#cars.print_info()
		#print "--"*20
		
		self.update_car_data()

		if self.killed == self.population_size:
			self.next_generation()
			#time.sleep(1)

	def start(self):
		while True:
			self.update_car_data()
			if self.killed == self.population_size:
				self.next_generation()

	def update_car_data(self):
			for index,cars in enumerate(self.population_data):
				if not cars.isDead:
					cars.set_pos_and_vel([self.population[index][0].position.x,self.population[index][0].position.y],self.population[index][0].linearVelocity.x)
					if cars.isDead:
						#id you want to keep all the cars on the screen, (only for testing) commend the bottom 5 lines
						for wheel in self.population[index][1]:
							if wheel:
								self.world.DestroyBody(wheel) #remove wheels
						self.world.DestroyBody(self.population[index][0]) #remove chassis
						self.population[index] = None
						self.killed+=1  #turn this on only after all the mate,mutate methods work
						print "killed so far;",self.killed


	def sort_by_dist(self):
		self.population_data = sorted(self.population_data,key = lambda x:x.max_dist)
		self.leader_coors = [self.population_data[0].chassis.worldCenter.x,self.population_data[0].chassis.worldCenter.y]
		self.leader = self.population_data[0].chassis

	def check_dup(self,parent_pair,mates_list):
		dup = False
		if parent_pair in mates_list or parent_pair[::-1] in mates_list:
			dup = True
		return dup
	def get_parent_index(self,mates_list):
		parent1_index = random.randint(0,self.population_size-1)
		parent2_index = random.randint(0,self.population_size-1)
		while parent2_index == parent1_index and not self.check_dup([parent1_index,parent2_index],mates_list):
			parent1_index = random.randint(0,self.population_size-1)
			parent2_index = random.randint(0,self.population_size-1)
		return [parent1_index,parent2_index]


	def mate(self,parents):
		#mating the 2 cars works like this:
		#we take the info of the 2 parents as parent1 and parent2
		#we now define 2 swapPoints
		#if we have 10 attributes and the swappoints are 3 and 7
		#the child will have the following attribute array:
		#child = parent1[1..swapPoint1] + parent2[swapPoint1+1 ...swapPoint2] + parent1[swapPointswapPoint2...10]
		#atrribute list for our car: (also look at make_random_car() method in car class)
		#below is for 2 wheels
		#1.no.of wheels (1)
		#2.wheel_radius (2)
		#3.wheel_vertex (2)
		#4.wheel_density (2)
		#5.vertex_list   (8)
		#6.chassis_density (1)
		#total - 16 attributes [0..15]
		total_attributes = 15
		attribute_index = 0
		parents = [self.population_data[parents[0]].car_def,self.population_data[parents[1]].car_def]
		swap_point1 = random.randint(0,total_attributes)
		swap_point2 = random.randint(0,total_attributes)
		while swap_point1 == swap_point2:
			swap_point2 = random.randint(0,total_attributes)

		child = car_info()
		curr_parent = 0
		child.set_wheel_count = parents[curr_parent].get_wheel_count()
		attribute_index += 1
		curr_parent = self.which_parent(attribute_index,curr_parent,swap_point1,swap_point2)
		#print "cross over points,",swap_point1,"<->",swap_point2
		#print "wheel radius corssings"
		for i in range(child.get_wheel_count()):
			#print "now takign attribute from parent",curr_parent
			child.wheel_radius[i] = parents[curr_parent].wheel_radius[i]
			attribute_index += 1
			curr_parent = self.which_parent(attribute_index,curr_parent,swap_point1,swap_point2)
			#print "new curr parent is ",curr_parent
		#print "wheel vertex corssings"
		for i in range(child.get_wheel_count()):
			#print "now takign attribute from parent",curr_parent
			child.wheel_vertex[i] = parents[curr_parent].wheel_vertex[i]
			attribute_index += 1
			curr_parent = self.which_parent(attribute_index,curr_parent,swap_point1,swap_point2)
		#print "wheel density corssings"
		for i in range(child.get_wheel_count()):
			#print "now takign attribute from parent",curr_parent
			child.wheel_density[i] = parents[curr_parent].wheel_density[i]
			attribute_index += 1
			curr_parent = self.which_parent(attribute_index,curr_parent,swap_point1,swap_point2)
		#print "vertex list corssings"
		for i in range(len(child.get_vertex_list())):
			#print "now takign attribute from parent",curr_parent
			child.vertex_list[i] = parents[curr_parent].vertex_list[i]
			attribute_index += 1
			curr_parent = self.which_parent(attribute_index,curr_parent,swap_point1,swap_point2)
		
		child.set_chassis_density = parents[curr_parent].get_chassis_density()
		#print "attributes completed:",attribute_index
		child_car = car(self.world,random = False,car_def = child)
		return child_car

	def which_parent(self,index,last_parent,swp1,swp2):
		if index == swp1 or index == swp2:
			#print "changed parent:",abs(last_parent-1)
			#return abs(last_parent-1) #if 0, 1 is returned|| if 1, 0 is returned
			if last_parent == 0:
				return 1
			else:
				return 0
		else:
			return last_parent

	#def mutate(self,child):


	def next_generation(self):
		self.sort_by_dist()
		n = 3
		#take the top n contenders of the prev generation and append it to the new list
		#and mate 2 random parents (population_size-n) times and append their children to the new list
		#now replace the exisitng poulation with the new population
		new_population = []
		new_population_data = []
		for i in range(n):
			new_car = car(self.world,random=False,car_def = self.population_data[i].car_def)
			new_population.append([new_car.chassis,new_car.wheels])
			new_population_data.append(car_data(self.population_data[i].chassis,self.population_data[i].wheels,self.population_data[i].car_def))


		mates_list = [] #pairs of indices of parents that have mated (to avoid duplicates in the same generation)
		while len(new_population) < self.population_size:
			parents = self.get_parent_index(mates_list)
			mates_list.append(parents)
			child = self.mate(parents) #we're passing only the indices of the parents
			#child = self.mutate(child.car_def)
			new_population.append([child.chassis,child.wheels])
			new_population_data.append(car_data(child.chassis,child.wheels,child.car_def))

		#change generation.
		print len(new_population)
		print "START NEW GENERATION!!!!!!!!!"

		self.killed = 0
		for index,elem in enumerate(new_population_data):
			self.population_data[index] = elem

		for index,elem in enumerate(new_population):
			self.population[index] = elem

		self.sort_by_dist()
		#self.population = new_population
		#self.population_data = new_population_data
		#start the drawing again and the loop continues.....


	def create_generation_1(self):
		for i in range(self.population_size):
			temp  = car(self.world)
			self.population.append([temp.get_car_chassis(),temp.get_car_wheels()])
			self.population_data.append(car_data(temp.get_car_chassis(),temp.get_car_wheels(),temp.car_def))

#if __name__ == "__main__":
#	main(do_stuff)
m = do_stuff()

#few problems:

#if there is a huge difference between the density of wheel and the dnsity of chassis, then
#the wheel will literally drag the car.. i.e wheel running forward and chassis being 
#dragged (connected by a anchor)
#so adjust the density of wheel and chassis (in global) accordingly - DONE

#check the terrain generation code, its very jagged.. - DONE

#new problem - trying to get the camera to focus on the current leader - DONE
#have another method where we sort the currently alive cars by their distance AND their dead or alive status
#let the sort criteria be - dist+(if dead:-999999 else 0) and then sort and get the highest [x,y] and add it to camera as offset