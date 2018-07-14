#coding:utf-8

import pyglet

music = pyglet.resource.media('music.mp3')
music.play()

pyglet.app.run()

'''
sound = pyglet.resource.media('shot.wav', streaming=False)
sound.play()
'''