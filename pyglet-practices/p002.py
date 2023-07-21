import pyglet
from pyglet.window import key
from pyglet.app import run
from pyglet.window import mouse

window = pyglet.window.Window(resizable=True)

@window.event
def on_resize(width, height):
    print(f'The window was resized to {width},{height}')

@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.A:
        print('The "A" key was pressed.')
    elif symbol == key.LEFT:
        print('The left arrow key was pressed.')
    elif symbol == key.ENTER:
        print('The enter key was pressed.')

@window.event
def on_mouse_press(x, y, button, modifiers):
    if button == mouse.LEFT:
        print('The left mouse button was pressed.')


@window.event
def on_draw():
    window.clear()

run()


