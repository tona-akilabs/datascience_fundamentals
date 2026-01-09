from math import exp

def interest():
    p = 100
    r = .20
    t = 2.0
    n = 12

    a = p * (1 + (r/n))**(n * t)
    print(a) # prints 148.69146179463576

def interest2():
    p = 100 # principal, starting amount
    r = .20 # interest rate, by year
    t = 2.0 # time, number of years
    a = p * exp(r*t)
    print(a) # prints 149.18246976412703