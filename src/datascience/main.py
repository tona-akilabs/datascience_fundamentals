import datascience.simple as helper1
from sympy import symbols
from sympy.plotting import plot3d

def main():
    print("Hello World")
    print(2**3) # prints 8
    helper1.interest()
    helper1.interest2()

    x, y = symbols('x y')
    f = 2 * x + 3 * y

    plot3d(f)
    #plot3d(f, (x, -5, 5), (y, -5, 5))
