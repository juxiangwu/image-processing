#coding:utf-8
import pyglet
from pyglet.window import key
from pyglet.window import mouse
import os
cwd = os.getcwd()
datas = os.path.abspath(os.path.join(cwd,'datas'))
pyglet.resource.path = [datas]
pyglet.resource.reindex()
window = pyglet.window.Window()
image = pyglet.resource.image('bird.jpg')
# window.push_handlers(pyglet.window.event.WindowEventLogger())
@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.A:
        print('The "A" key was pressed.')
    elif symbol == key.LEFT:
        print('The left arrow key was pressed.')
    elif symbol == key.ENTER:
        print('The enter key was pressed.')
    elif symbol == key.Q:
        exit(0)

@window.event
def on_mouse_press(x, y, button, modifiers):
    if button == mouse.LEFT:
        print('The left mouse button was pressed.')

@window.event
def on_draw():
    window.clear()
    image.blit(0, 0)

pyglet.app.run()