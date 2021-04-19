
import tkinter as tk
import os



def find_files():
    for (path, dir, files) in os.walk("c:/"):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.py':
            print("%s/%s" % (path, filename))
    return


class aaa():
    def __init__(self):
        self.a = 'hello'
    def print_hello(self):
        b = 'world~!!'
        print(self.a+', '+b)
    

if __name__ == '__main__':
    print('hello, world~!')
    temp = aaa()
    temp.print_hello()



