import numpy as np


# Code provided by: 
# S. De Marchi, F. Marchetti, E. Perracchione, D. Poggiali, Polynomial interpolation via mapped bases without resampling,
# JCAM. https://github.com/pog87/FakeNodes
def fakenodes_interp(x,y,xx,S=None, degree=None):
    if S is None:
        #perform tradictional interpolation
        S=lambda x:x
    if degree is None:
        degree=len(x)-1
    #
    x_=S(x); xx_ = S(xx)
    P=np.polyfit(x_,y,degree)
    yy=np.polyval(P,xx_)
    #yy=lagrange_interp(xx_,x_,y)
    return yy