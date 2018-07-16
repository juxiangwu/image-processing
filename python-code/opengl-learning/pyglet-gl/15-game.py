# -*- coding: utf-8 -*-
#
#  spacegame.py
#  
#  Copyright (C) 2014 Guillermo Aguilar <gmo.aguilar.c@gmail.com>
#  
#  Based partially on PyGletSpace.py
#  Copyright (C) 2007 Mark Mruss <selsine@gmail.com>
#  http://www.learningpython.com
#
#  SpaceGame is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  SpaceGame is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with SpaceGame.  If not, see <http://www.gnu.org/licenses/>

import random, math
import pyglet
from pyglet import window
from pyglet import clock
from pyglet import font
from pyglet.window import key 


def distance(a,b):
    """
    returns the euclidean distance
    """
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
    
###############################################################################
class SpaceGame(window.Window):
    
    def __init__(self, *args, **kwargs):

        #Let all of the arguments pass through
        self.win = window.Window.__init__(self, *args, **kwargs)
        
        self.maxaliens = 50 # max num of aliens simultaneously on the screen
        
        clock.schedule_interval(self.create_alien, 0.5)
        clock.schedule_interval(self.update, 1.0/30) # update at FPS of Hz
        
        #clock.set_fps_limit(30)
        
        # setting text objects
        ft = font.load('Tahoma', 20)    #Create a font for our FPS clock
        self.fpstext = font.Text(ft, y=10)   # object to display the FPS
        self.score = font.Text(ft, x=self.width, y=self.height, 
                               halign=pyglet.font.Text.RIGHT, 
                               valign=pyglet.font.Text.TOP)
        
        # reading and saving images
        self.spaceship_image = pyglet.image.load('datas/game-resources/ship3.png')
        self.alien_image = pyglet.image.load('datas/game-resources/invader.png')
        self.bullet_image = pyglet.image.load('datas/game-resources/bullet_white_16.png')
        
        # create one spaceship
        self.spaceship = Spaceship(self.spaceship_image, x=50, y=50)
        
        self.aliens=[] # list of Alien objects
        self.bullets=[] # list of Bullet objects
        
    def create_alien(self, dt):
        
        if len(self.aliens) < self.maxaliens:
            self.aliens.append( Alien(self.alien_image, 
                                      x=random.randint(0, self.width) , 
                                      y=self.height))
        
        
    def update(self, dt):
        
        # updating aliens
        for alien in self.aliens:
            alien.update()
            if alien.dead:
                self.aliens.remove(alien)
        
        # updating bullets
        for bullet in self.bullets:
            bullet.update()
            if bullet.dead:
                self.bullets.remove(bullet)
        
        # checking for bullet - alien collision
        for bullet in self.bullets:
            for alien in self.aliens:
                if distance(alien, bullet) < (alien.width/2 + bullet.width/2):
                    bullet.on_kill()                    
                    bullet.dead=True
                    alien.dead=True
        
        # checking ship - alien collision
        for alien in self.aliens:
            if distance(alien, self.spaceship) < (alien.width/2+ self.spaceship.width/2):
                print("Game Over")
                self.dispatch_event('on_close') 


    def on_draw(self):
        self.clear() # clearing buffer
        clock.tick() # ticking the clock
        
        # showing FPS
        self.fpstext.text = "fps: %d" % clock.get_fps()
        self.fpstext.draw()
        
        self.score.text = "# Killed: %d" % self.spaceship.kills
        self.score.draw()
        
        # drawing objects of the game
        self.spaceship.draw()
        for alien in self.aliens:
            alien.draw()
        for bullet in self.bullets:
            bullet.draw()
            
        # flipping
        self.flip()
    
    
    ## Event handlers
    def on_key_press(self, symbol, modifiers):
        
        if symbol == key.ESCAPE:
            self.dispatch_event('on_close')  
            
        elif symbol == key.LEFT:
            self.spaceship.x -= 10
        elif symbol == key.RIGHT:
            self.spaceship.x += 10
            
    def on_mouse_motion(self, x, y, dx, dy):
        self.spaceship.x = x

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.spaceship.x = x
        
    def on_mouse_press(self, x, y, button, modifiers):

        if (button == 1):
            self.bullets.append(Bullet(self.bullet_image, self.spaceship, 
                                       self.height, 
                                       x=self.spaceship.x + self.spaceship.width/2.0,
                                       y=self.spaceship.y + self.spaceship.height/2.0))



####################################################################
class Spaceship(pyglet.sprite.Sprite):

    def __init__(self, *args, **kwargs):
        pyglet.sprite.Sprite.__init__(self, *args, **kwargs)
        self.kills = 0
        
    def on_kill(self):
        self.kills += 1


###################################################################    
class Alien(pyglet.sprite.Sprite):

    def __init__(self, *args, **kwargs):
        pyglet.sprite.Sprite.__init__(self, *args, **kwargs)
        self.y_velocity = 5
        self.set_x_velocity()
        self.x_move_count = 0
        self.dead = False

    def set_x_velocity(self):
        self.x_velocity = random.randint(-3,3)

    def update(self):
        # update position
        self.y -= self.y_velocity
        self.x += self.x_velocity
        # and counter
        self.x_move_count += 1
        
        #Have we gone beneath the botton of the screen?
        if (self.y < 0):
            self.dead = True
            
        if (self.x_move_count >=30):
            self.x_move_count = 0
            self.set_x_velocity()
   
##################################################################    
class Bullet(pyglet.sprite.Sprite):

    def __init__(self, image_data, parent_ship, screen_top, **kwargs):
        pyglet.sprite.Sprite.__init__(self, image_data, **kwargs)
        self.dead = False
        self.velocity = 5
        self.screen_top = screen_top
        self.parent_ship = parent_ship
        

    def update(self):
        self.y += self.velocity
        if (self.y > self.screen_top):
            self.dead = True

    def on_kill(self):
        self.parent_ship.on_kill()    
        # when hitting an alien, call on_kill to update spaceship counter
   
###################################################################
  
if __name__ == "__main__":
    win = SpaceGame(caption="Aliens!! Invaders from Space!!", height=600, width=800)
    pyglet.app.run()