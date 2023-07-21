import pyglet

window = pyglet.window.Window()
image = pyglet.resource.image('kitten.jpg')
# image = pyglet.image.load('kitten.jpg')

print(window)
print(image)

image.width = window.width*0.8
image.height = window.height*0.8

# sprite = pyglet.sprite.Sprite(image)

@window.event
def on_draw():
    window.clear()
    image.blit(0,0)
    # sprite.draw()

pyglet.app.run()






