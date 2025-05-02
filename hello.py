# This is a comment
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

import requests

def divide(a, b):
    return a / b


def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(-1, -1) == -2
    assert add(0, 0) == 0

if __name__ == "__main__":
    print("hello world")
    test_add()