# -*- coding: utf-8 -*-
"""
-----------------------------------------------------
  Modified by Kaan Cökerim¹ on 17. November 2025

  Added a vectorized `atanh` routine and changed
  the use of the `sympy`-package to `numpy` to allow
  for vectorization.

  ¹Tectonic Geodesy, Ruhr University Bochum, Germany
  Email: kaan.coekerim@rub.de

  Original authors:
     Sylvain Barbot (sbarbot@ntu.edu.sg),
     Qiu Qiang (qiuqiang2012@gmail.com)
     on April 12, 2017 (see below for details)
-----------------------------------------------------
"""
"""
    % function COMPUTEDISPLACEMENTVERTICALSHEARZONE computes the displacement
    % field associated with deforming vertical shear zones using the analytic
    % solution considering the following geometry.
    %
    %                      N (x1)
    %                     /
    %                    /| strike (theta)          E (x2)
    %        q1,q2,q3 ->@--------------------------+
    %                   |                        w |     +
    %                   |                        i |    /
    %                   |                        d |   / s
    %                   |                        t |  / s
    %                   |                        h | / e
    %                   |                          |/ n
    %                   +--------------------------+  k
    %                   :       l e n g t h       /  c
    %                   |                        /  i
    %                   :                       /  h
    %                   |                      /  t
    %                   :                     /
    %                   |                    +
    %                   Z (x3)
    %
    %
    % Input:
    % x1, x2, x3         northing, easting, and depth of the observation point,
    % q1, q2, q3         north, east and depth coordinates of the shear zone,
    % L, T, W            length, thickness, and width of the shear zone,
    % theta (degree)     strike of the shear zone,
    % epsvijp            source strain component 11, 12, 13, 22, 23 and 33
    %                    in the shear zone in the system of reference tied to
    %                    the shear zone,
    % G, nu              shear modulus and Poisson's ratio in the half space.
    %
    % Output:
    % u1                 displacement component in the north direction,
    % u2                 displacement component in the east direction,
    % u3                 displacement component in the down direction.
    %
    % Author: Sylvain Barbot (sbarbot@ntu.edu.sg) -
    % Translated for python by Qiu Qiang (qiuqiang2012@gmail.com) - April 12, 2017

"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cmcrameri.cm as cm


def computeDisplacementVerticalShearZone(x1,x2,x3,
                                         q1,q2,q3,
                                         L,T,W,theta,
                                         epsv11p,epsv12p,epsv13p,epsv22p,epsv23p,epsv33p,
                                         G,nu
                                         ):
    #  Lame parameter
    Lambda=G*2*nu/(1-2*nu)

    # isotropic strain
    epsvkk=epsv11p+epsv22p+epsv33p

    #  rotate observation points to the shear-zone-centric system of coordinates
    theta = np.deg2rad(theta)
    t1= (x1-q1)*np.cos(theta)+(x2-q2)*np.sin(theta)
    x2=-(x1-q1)*np.sin(theta)+(x2-q2)*np.cos(theta)
    x1=t1


    # Greens' function
    r1 = lambda y1,y2,y3: np.sqrt((x1-y1)**2+(x2-y2)**2+(x3-y3)**2)
    r2 = lambda y1,y2,y3: np.sqrt((x1-y1)**2+(x2-y2)**2+(x3+y3)**2)

    pi = np.pi

    J1112=lambda y1,y2,y3: \
    (1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(2*r2(y1,y2,y3)**(-1)*x3*( \
      x1-y1)*(x2-y2)*y3*((x1-y1)**2+(x3+y3)**2)**( \
      -1)-4*((-1)+nu)*((-1)+2*nu)*(x3+y3)*np.arctan((x1-y1)**( \
      -1)*(x2-y2))-x3*np.arctan2(x3,x1-y1)-3*x3* \
      np.arctan2(3*x3,x1-y1)+4*nu*x3*np.arctan2(-nu*x3,x1- \
      y1)+4*((-1)+nu)*((-1)+2*nu)*(x3+y3)*np.arctan2(r2(y1,y2,y3)*(-x1+y1),( \
      x2-y2)*(x3+y3))-4*((-1)+nu)*(x3-y3)*np.arctan2(r1(y1,y2,y3)*( \
      x3-y3),(x1-y1)*(x2-y2))+3*y3*np.arctan2((-3)*y3, \
      x1-y1)-y3*np.arctan2(y3,x1-y1)-4*nu*y3*np.arctan2( \
      nu*y3,x1-y1)-4*((-1)+nu)*(x3+y3)*np.arctan2(r2(y1,y2,y3)*(x3+y3),( \
      x1-y1)*(x2-y2))+xLogy(-((-3)+4*nu)*(x1- \
      y1),r1(y1,y2,y3)+x2-y2)+xLogy((5+4*nu*((-3)+2*nu))*(x1-y1), \
      r2(y1,y2,y3)+x2-y2)+xLogy((-4)*((-1)+nu)*(x2-y2),r1(y1,y2,y3)+x1- \
      y1)+xLogy((-4)*((-1)+nu)*(x2-y2),r2(y1,y2,y3)+x1-y1))

    J1113=lambda y1,y2,y3: \
    (1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(2*r2(y1,y2,y3)**(-1)*(x1+( \
      -1)*y1)*((x1-y1)**2+(x2-y2)**2)**(-1)*(-((-1)+ \
      nu)*((-1)+2*nu)*r2(y1,y2,y3)**2*(x3+y3)+((-1)+nu)*((-1)+2*nu)*r2(y1,y2,y3)* \
      y3*(2*x3+y3)+x3*((x1-y1)**2+(x2-y2)**2+x3*(x3+y3)) \
      )+x2*np.arctan2(-x2,x1-y1)-3*x2*np.arctan2(3*x2,x1- \
      y1)+4*nu*x2*np.arctan2(-nu*x2,x1-y1)-4*((-1)+nu)*( \
      x2-y2)*np.arctan2(r1(y1,y2,y3)*(x2-y2),(x1-y1)*(x3-y3) \
      )+4*((-1)+nu)*(x2-y2)*np.arctan2(r2(y1,y2,y3)*(x2-y2),(x1- \
      y1)*(x3+y3))+3*y2*np.arctan2((-3)*y2,x1-y1)-y2*np.arctan2( \
      y2,x1-y1)-4*nu*y2*np.arctan2(nu*y2,x1-y1)+xLogy((-1) \
      *((-3)+4*nu)*(x1-y1),r1(y1,y2,y3)+x3-y3)+xLogy(-(3 \
      -6*nu+4*nu**2)*(x1-y1),r2(y1,y2,y3)+x3+y3)+xLogy((-4)*((-1)+nu)*( \
      x3-y3),r1(y1,y2,y3)+x1-y1)+xLogy(4*((-1)+nu)*(x3+y3),r2(y1,y2,y3)+x1+( \
      -1)*y1))

    J1123=lambda y1,y2,y3: \
    (1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*((-2)*r2(y1,y2,y3)**(-1)*(( \
      x1-y1)**2+(x2-y2)**2)**(-1)*(x2-y2)*((x1+(-1) \
      *y1)**2+(x3+y3)**2)**(-1)*(x3*((x3**2+(x1-y1)**2)*( \
      x3**2+(x1-y1)**2+(x2-y2)**2)+x3*(3*x3**2+2*(x1+(-1) \
      *y1)**2+(x2-y2)**2)*y3+3*x3**2*y3**2+x3*y3**3)-(( \
      -1)+nu)*((-1)+2*nu)*r2(y1,y2,y3)**2*(x3+y3)*((x1-y1)**2+(x3+y3) \
      **2)+((-1)+nu)*((-1)+2*nu)*r2(y1,y2,y3)*y3*(2*x3+y3)*((x1-y1) \
      **2+(x3+y3)**2))+2*((-1)+nu)*((-1)+2*nu)*(x1-y1)*np.arctan(( \
      x1-y1)*(x2-y2)**(-1))+x1*np.arctan2(-x1,x2-y2) \
      -3*x1*np.arctan2(3*x1,x2-y2)+4*nu*x1*np.arctan2(-nu*x1, \
      x2-y2)+3*y1*np.arctan2((-3)*y1,x2-y2)-y1*np.arctan2( \
      y1,x2-y2)-4*nu*y1*np.arctan2(nu*y1,x2-y2)+2*((-1)+ \
      2*nu)*(x1-y1)*np.arctan2(r1(y1,y2,y3)*(-x1+y1),(x2-y2)*(x3+ \
      (-1)*y3))+2*(1-2*nu)**2*(x1-y1)*np.arctan2(r2(y1,y2,y3)*(-x1+ \
      y1),(x2-y2)*(x3+y3))+xLogy((-2)*x3,r2(y1,y2,y3)-x2+y2)+xLogy(( \
      -1)*((-3)+4*nu)*(x2-y2),r1(y1,y2,y3)+x3-y3)+xLogy(-(3+( \
      -6)*nu+4*nu**2)*(x2-y2),r2(y1,y2,y3)+x3+y3)+xLogy(-((-3)+4* \
      nu)*(x3-y3),r1(y1,y2,y3)+x2-y2)+xLogy(-(5+4*nu*((-3)+2* \
      nu))*(x3+y3),r2(y1,y2,y3)+x2-y2))

    J2112=lambda y1,y2,y3: \
    (1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(-r1(y1,y2,y3)+(1+8*(( \
      -1)+nu)*nu)*r2(y1,y2,y3)-2*r2(y1,y2,y3)**(-1)*x3*y3+xLogy((-4)*((-1)+nu)*(( \
      -1)+2*nu)*(x3+y3),r2(y1,y2,y3)+x3+y3))


    J2113=lambda y1,y2,y3: \
    (1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(2*r2(y1,y2,y3)**(-1)*((x1+ \
      (-1)*y1)**2+(x2-y2)**2)**(-1)*(x2-y2)*(-((-1)+ \
      nu)*((-1)+2*nu)*r2(y1,y2,y3)**2*(x3+y3)+((-1)+nu)*((-1)+2*nu)*r2(y1,y2,y3)* \
      y3*(2*x3+y3)+x3*((x1-y1)**2+(x2-y2)**2+x3*(x3+y3)) \
      )+xLogy(-((-1)-2*nu+4*nu**2)*(x2-y2),r2(y1,y2,y3)+x3+y3)+ \
      xLogy(-x2+y2,r1(y1,y2,y3)+x3-y3))


    J2123=lambda y1,y2,y3: \
    (1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(2*r2(y1,y2,y3)**(-1)*(x1+( \
      -1)*y1)*((x1-y1)**2+(x2-y2)**2)**(-1)*(-((-1)+ \
      nu)*((-1)+2*nu)*r2(y1,y2,y3)**2*(x3+y3)+((-1)+nu)*((-1)+2*nu)*r2(y1,y2,y3)* \
      y3*(2*x3+y3)+x3*((x1-y1)**2+(x2-y2)**2+x3*(x3+y3)) \
      )+xLogy(-((-1)-2*nu+4*nu**2)*(x1-y1),r2(y1,y2,y3)+x3+y3)+ \
      xLogy(-x1+y1,r1(y1,y2,y3)+x3-y3))


    J3112=lambda y1,y2,y3: \
    (-1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*((-2)*r2(y1,y2,y3)**(-1)* \
      x3*(x2-y2)*y3*(x3+y3)*((x1-y1)**2+(x3+y3)**2)**( \
      -1)+4*((-1)+nu)*((-1)+2*nu)*(x1-y1)*np.arctan((x1-y1) \
      *(x2-y2)**(-1))+4*((-1)+nu)*((-1)+2*nu)*(x1-y1)* \
      np.arctan2(r2(y1,y2,y3)*(-x1+y1),(x2-y2)*(x3+y3))+xLogy((-4)*((-1)+ \
      nu)*((-1)+2*nu)*(x2-y2),r2(y1,y2,y3)+x3+y3)+xLogy(x3-y3,r1(y1,y2,y3)+ \
      x2-y2)+xLogy(-x3-7*y3-8*nu**2*(x3+y3)+8*nu*( \
      x3+2*y3),r2(y1,y2,y3)+x2-y2))


    J3113=lambda y1,y2,y3: \
    (-1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(r1(y1,y2,y3)+((-1)-8*(( \
      -1)+nu)*nu)*r2(y1,y2,y3)-2*r2(y1,y2,y3)**(-1)*x3*y3+2*((-3)+4*nu)*x3* \
      acoth(r2(y1,y2,y3)**(-1)*(x3+y3))+xLogy(2*(3*x3+2*y3-6*nu*(x3+y3)+ \
      4*nu**2*(x3+y3)),r2(y1,y2,y3)+x3+y3))


    J3123=lambda y1,y2,y3: \
    (-1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(2*r2(y1,y2,y3)**(-1)*x3*( \
      x1-y1)*(x2-y2)*y3*((x1-y1)**2+(x3+y3)**2)**( \
      -1)+4*((-1)+nu)*((-1)+2*nu)*(x3+y3)*np.arctan((x1-y1)**(-1) \
      *(x2-y2))+4*((-1)+2*nu)*(nu*x3+((-1)+nu)*y3)*np.arctan2( \
      r2(y1,y2,y3)*(x1-y1),(x2-y2)*(x3+y3))+xLogy(x1-y1,r1(y1,y2,y3)+x2+ \
      (-1)*y2)+xLogy(-(1+8*((-1)+nu)*nu)*(x1-y1),r2(y1,y2,y3)+x2+( \
      -1)*y2))


    J1212=lambda y1,y2,y3: \
    (1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(-r1(y1,y2,y3)+(1+8*(( \
      -1)+nu)*nu)*r2(y1,y2,y3)-2*r2(y1,y2,y3)**(-1)*x3*y3+xLogy((-4)*((-1)+nu)*(( \
      -1)+2*nu)*(x3+y3),r2(y1,y2,y3)+x3+y3))


    J1213=lambda y1,y2,y3: \
    (1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(2*r2(y1,y2,y3)**(-1)*((x1+ \
      (-1)*y1)**2+(x2-y2)**2)**(-1)*(x2-y2)*(-((-1)+ \
      nu)*((-1)+2*nu)*r2(y1,y2,y3)**2*(x3+y3)+((-1)+nu)*((-1)+2*nu)*r2(y1,y2,y3)* \
      y3*(2*x3+y3)+x3*((x1-y1)**2+(x2-y2)**2+x3*(x3+y3)) \
      )+xLogy(-((-1)-2*nu+4*nu**2)*(x2-y2),r2(y1,y2,y3)+x3+y3)+ \
      xLogy(-x2+y2,r1(y1,y2,y3)+x3-y3))


    J1223=lambda y1,y2,y3: \
    (1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(2*r2(y1,y2,y3)**(-1)*(x1+( \
      -1)*y1)*((x1-y1)**2+(x2-y2)**2)**(-1)*(-((-1)+ \
      nu)*((-1)+2*nu)*r2(y1,y2,y3)**2*(x3+y3)+((-1)+nu)*((-1)+2*nu)*r2(y1,y2,y3)* \
      y3*(2*x3+y3)+x3*((x1-y1)**2+(x2-y2)**2+x3*(x3+y3)) \
      )+xLogy(-((-1)-2*nu+4*nu**2)*(x1-y1),r2(y1,y2,y3)+x3+y3)+ \
      xLogy(-x1+y1,r1(y1,y2,y3)+x3-y3))


    J2212=lambda y1,y2,y3: \
    (1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(2*r2(y1,y2,y3)**(-1)*x3*( \
      x1-y1)*(x2-y2)*y3*((x2-y2)**2+(x3+y3)**2)**( \
      -1)-4*((-1)+nu)*((-1)+2*nu)*(x3+y3)*np.arctan((x1-y1)*( \
      x2-y2)**(-1))-x3*np.arctan2(x3,x1-y1)-3*x3* \
      np.arctan2(3*x3,x1-y1)+4*nu*x3*np.arctan2(-nu*x3,x1- \
      y1)+4*((-1)+nu)*((-1)+2*nu)*(x3+y3)*np.arctan2(r2(y1,y2,y3)*(-x2+y2),( \
      x1-y1)*(x3+y3))-4*((-1)+nu)*(x3-y3)*np.arctan2(r1(y1,y2,y3)*( \
      x3-y3),(x1-y1)*(x2-y2))+3*y3*np.arctan2((-3)*y3, \
      x1-y1)-y3*np.arctan2(y3,x1-y1)-4*nu*y3*np.arctan2( \
      nu*y3,x1-y1)-4*((-1)+nu)*(x3+y3)*np.arctan2(r2(y1,y2,y3)*(x3+y3),( \
      x1-y1)*(x2-y2))+xLogy((-4)*((-1)+nu)*(x1-y1), \
      r1(y1,y2,y3)+x2-y2)+xLogy((-4)*((-1)+nu)*(x1-y1),r2(y1,y2,y3)+x2- \
      y2)+xLogy(-((-3)+4*nu)*(x2-y2),r1(y1,y2,y3)+x1-y1)+xLogy( \
      (5+4*nu*((-3)+2*nu))*(x2-y2),r2(y1,y2,y3)+x1-y1))


    J2213=lambda y1,y2,y3: \
    (1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*((-2)*r2(y1,y2,y3)**(-1)*( \
      x1-y1)*((x1-y1)**2+(x2-y2)**2)**(-1)*((x2+(-1) \
      *y2)**2+(x3+y3)**2)**(-1)*(x3*((x3**2+(x2-y2)**2)*( \
      x3**2+(x1-y1)**2+(x2-y2)**2)+x3*(3*x3**2+(x1- \
      y1)**2+2*(x2-y2)**2)*y3+3*x3**2*y3**2+x3*y3**3)-( \
      (-1)+nu)*((-1)+2*nu)*r2(y1,y2,y3)**2*(x3+y3)*((x2-y2)**2+(x3+y3) \
      **2)+((-1)+nu)*((-1)+2*nu)*r2(y1,y2,y3)*y3*(2*x3+y3)*((x2-y2) \
      **2+(x3+y3)**2))+2*((-1)+nu)*((-1)+2*nu)*(x2-y2)*np.arctan(( \
      x1-y1)**(-1)*(x2-y2))+x2*np.arctan2(-x2,x1-y1) \
      -3*x2*np.arctan2(3*x2,x1-y1)+4*nu*x2*np.arctan2(-nu*x2, \
      x1-y1)+3*y2*np.arctan2((-3)*y2,x1-y1)-y2*np.arctan2( \
      y2,x1-y1)-4*nu*y2*np.arctan2(nu*y2,x1-y1)+2*((-1)+ \
      2*nu)*(x2-y2)*np.arctan2(r1(y1,y2,y3)*(-x2+y2),(x1-y1)*(x3+ \
      (-1)*y3))+2*(1-2*nu)**2*(x2-y2)*np.arctan2(r2(y1,y2,y3)*(-x2+ \
      y2),(x1-y1)*(x3+y3))+xLogy((-2)*x3,r2(y1,y2,y3)-x1+y1)+xLogy(( \
      -1)*((-3)+4*nu)*(x1-y1),r1(y1,y2,y3)+x3-y3)+xLogy(-(3+( \
      -6)*nu+4*nu**2)*(x1-y1),r2(y1,y2,y3)+x3+y3)+xLogy(-((-3)+4* \
      nu)*(x3-y3),r1(y1,y2,y3)+x1-y1)+xLogy(-(5+4*nu*((-3)+2* \
      nu))*(x3+y3),r2(y1,y2,y3)+x1-y1))


    J2223=lambda y1,y2,y3: \
    (1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(2*r2(y1,y2,y3)**(-1)*((x1+ \
      (-1)*y1)**2+(x2-y2)**2)**(-1)*(x2-y2)*(-((-1)+ \
      nu)*((-1)+2*nu)*r2(y1,y2,y3)**2*(x3+y3)+((-1)+nu)*((-1)+2*nu)*r2(y1,y2,y3)* \
      y3*(2*x3+y3)+x3*((x1-y1)**2+(x2-y2)**2+x3*(x3+y3)) \
      )+x1*np.arctan2(-x1,x2-y2)-3*x1*np.arctan2(3*x1,x2- \
      y2)+4*nu*x1*np.arctan2(-nu*x1,x2-y2)-4*((-1)+nu)*( \
      x1-y1)*np.arctan2(r1(y1,y2,y3)*(x1-y1),(x2-y2)*(x3-y3) \
      )+4*((-1)+nu)*(x1-y1)*np.arctan2(r2(y1,y2,y3)*(x1-y1),(x2- \
      y2)*(x3+y3))+3*y1*np.arctan2((-3)*y1,x2-y2)-y1*np.arctan2( \
      y1,x2-y2)-4*nu*y1*np.arctan2(nu*y1,x2-y2)+xLogy((-1) \
      *((-3)+4*nu)*(x2-y2),r1(y1,y2,y3)+x3-y3)+xLogy(-(3 \
      -6*nu+4*nu**2)*(x2-y2),r2(y1,y2,y3)+x3+y3)+xLogy((-4)*((-1)+nu)*( \
      x3-y3),r1(y1,y2,y3)+x2-y2)+xLogy(4*((-1)+nu)*(x3+y3),r2(y1,y2,y3)+x2+( \
      -1)*y2))


    J3212=lambda y1,y2,y3: \
    (-1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*((-2)*r2(y1,y2,y3)**(-1)* \
      x3*(x1-y1)*y3*(x3+y3)*((x2-y2)**2+(x3+y3)**2)**( \
      -1)+4*((-1)+nu)*((-1)+2*nu)*(x2-y2)*np.arctan((x1-y1) \
      **(-1)*(x2-y2))+4*((-1)+nu)*((-1)+2*nu)*(x2-y2)* \
      np.arctan2(r2(y1,y2,y3)*(-x2+y2),(x1-y1)*(x3+y3))+xLogy((-4)*((-1)+ \
      nu)*((-1)+2*nu)*(x1-y1),r2(y1,y2,y3)+x3+y3)+xLogy(x3-y3,r1(y1,y2,y3)+ \
      x1-y1)+xLogy(-x3-7*y3-8*nu**2*(x3+y3)+8*nu*( \
      x3+2*y3),r2(y1,y2,y3)+x1-y1))


    J3213=lambda y1,y2,y3: \
    (-1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(2*r2(y1,y2,y3)**(-1)*x3*( \
      x1-y1)*(x2-y2)*y3*((x2-y2)**2+(x3+y3)**2)**( \
      -1)+4*((-1)+nu)*((-1)+2*nu)*(x3+y3)*np.arctan((x1-y1)*(x2+( \
      -1)*y2)**(-1))+4*((-1)+2*nu)*(nu*x3+((-1)+nu)*y3)*np.arctan2( \
      r2(y1,y2,y3)*(x2-y2),(x1-y1)*(x3+y3))+xLogy(x2-y2,r1(y1,y2,y3)+x1+ \
      (-1)*y1)+xLogy(-(1+8*((-1)+nu)*nu)*(x2-y2),r2(y1,y2,y3)+x1+( \
      -1)*y1))


    J3223=lambda y1,y2,y3: \
    (-1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(r1(y1,y2,y3)+((-1)-8*(( \
      -1)+nu)*nu)*r2(y1,y2,y3)-2*r2(y1,y2,y3)**(-1)*x3*y3+2*((-3)+4*nu)*x3* \
      acoth(r2(y1,y2,y3)**(-1)*(x3+y3))+xLogy(2*(3*x3+2*y3-6*nu*(x3+y3)+ \
      4*nu**2*(x3+y3)),r2(y1,y2,y3)+x3+y3))


    J1312=lambda y1,y2,y3: \
    (-1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(2*r2(y1,y2,y3)**(-1)*x3*( \
      x2-y2)*y3*(x3+y3)*((x1-y1)**2+(x3+y3)**2)**(-1)+( \
      -4)*((-1)+nu)*((-1)+2*nu)*(x1-y1)*np.arctan((x1-y1)*( \
      x2-y2)**(-1))+4*((-1)+nu)*((-1)+2*nu)*(x1-y1)* \
      np.arctan2(r2(y1,y2,y3)*(x1-y1),(x2-y2)*(x3+y3))+xLogy(4*((-1)+nu) \
      *((-1)+2*nu)*(x2-y2),r2(y1,y2,y3)+x3+y3)+xLogy(x3-y3,r1(y1,y2,y3)+x2+( \
      -1)*y2)+xLogy((7+8*((-2)+nu)*nu)*x3+y3+8*((-1)+nu)*nu*y3, \
      r2(y1,y2,y3)+x2-y2))


    J1313=lambda y1,y2,y3: \
    (-1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(r1(y1,y2,y3)+r2(y1,y2,y3)**(-1)*((7+ \
      8*((-2)+nu)*nu)*r2(y1,y2,y3)**2+2*x3*y3)+2*((-3)+4*nu)*x3*acoth( \
      r2(y1,y2,y3)**(-1)*(x3+y3))+xLogy(2*((-3)*x3-2*y3+6*nu*(x3+y3) \
      -4*nu**2*(x3+y3)),r2(y1,y2,y3)+x3+y3))


    J1323=lambda y1,y2,y3: \
    (-1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*((-2)*r2(y1,y2,y3)**(-1)* \
      x3*(x1-y1)*(x2-y2)*y3*((x1-y1)**2+(x3+y3) \
      **2)**(-1)-4*((-1)+nu)*((-1)+2*nu)*(x3+y3)*np.arctan((x1- \
      y1)**(-1)*(x2-y2))-4*((-1)+nu)*((-3)*x3-y3+2* \
      nu*(x3+y3))*np.arctan2(r2(y1,y2,y3)*(x1-y1),(x2-y2)*(x3+y3))+ \
      xLogy(x1-y1,r1(y1,y2,y3)+x2-y2)+xLogy((7+8*((-2)+nu)*nu)*(x1+ \
      (-1)*y1),r2(y1,y2,y3)+x2-y2))


    J2312=lambda y1,y2,y3: \
    (-1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(2*r2(y1,y2,y3)**(-1)*x3*( \
      x1-y1)*y3*(x3+y3)*((x2-y2)**2+(x3+y3)**2)**(-1)+( \
      -4)*((-1)+nu)*((-1)+2*nu)*(x2-y2)*np.arctan((x1-y1)**( \
      -1)*(x2-y2))+4*((-1)+nu)*((-1)+2*nu)*(x2-y2)* \
      np.arctan2(r2(y1,y2,y3)*(x2-y2),(x1-y1)*(x3+y3))+xLogy(4*((-1)+nu) \
      *((-1)+2*nu)*(x1-y1),r2(y1,y2,y3)+x3+y3)+xLogy(x3-y3,r1(y1,y2,y3)+x1+( \
      -1)*y1)+xLogy((7+8*((-2)+nu)*nu)*x3+y3+8*((-1)+nu)*nu*y3, \
      r2(y1,y2,y3)+x1-y1))


    J2313=lambda y1,y2,y3: \
    (-1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*((-2)*r2(y1,y2,y3)**(-1)* \
      x3*(x1-y1)*(x2-y2)*y3*((x2-y2)**2+(x3+y3) \
      **2)**(-1)-4*((-1)+nu)*((-1)+2*nu)*(x3+y3)*np.arctan((x1- \
      y1)*(x2-y2)**(-1))-4*((-1)+nu)*((-3)*x3-y3+2* \
      nu*(x3+y3))*np.arctan2(r2(y1,y2,y3)*(x2-y2),(x1-y1)*(x3+y3))+ \
      xLogy(x2-y2,r1(y1,y2,y3)+x1-y1)+xLogy((7+8*((-2)+nu)*nu)*(x2+ \
      (-1)*y2),r2(y1,y2,y3)+x1-y1))


    J2323=lambda y1,y2,y3: \
    (-1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(r1(y1,y2,y3)+r2(y1,y2,y3)**(-1)*((7+ \
      8*((-2)+nu)*nu)*r2(y1,y2,y3)**2+2*x3*y3)+2*((-3)+4*nu)*x3*acoth( \
      r2(y1,y2,y3)**(-1)*(x3+y3))+xLogy(2*((-3)*x3-2*y3+6*nu*(x3+y3) \
      -4*nu**2*(x3+y3)),r2(y1,y2,y3)+x3+y3))


    J3312=lambda y1,y2,y3: \
    (1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(2*r2(y1,y2,y3)**(-1)*x3*( \
      x1-y1)*(x2-y2)*y3*((x1-y1)**2+(x3+y3)**2)**( \
      -1)*((x2-y2)**2+(x3+y3)**2)**(-1)*((x1-y1)**2+(x2+( \
      -1)*y2)**2+2*(x3+y3)**2)-3*x3*np.arctan2(3*x3,x1-y1) \
      -5*x3*np.arctan2(5*x3,x2-y2)+12*nu*x3*np.arctan2((-3)*nu*x3,x2+( \
      -1)*y2)+4*nu*x3*np.arctan2(-nu*x3,x1-y1)-8*nu**2* \
      x3*np.arctan2(nu**2*x3,x2-y2)+3*y3*np.arctan2((-3)*y3,x1- \
      y1)-5*y3*np.arctan2(5*y3,x2-y2)+12*nu*y3*np.arctan2((-3)* \
      nu*y3,x2-y2)-4*nu*y3*np.arctan2(nu*y3,x1-y1)-8* \
      nu**2*y3*np.arctan2(nu**2*y3,x2-y2)+2*((-1)+2*nu)*(x3+(-1) \
      *y3)*np.arctan2(r1(y1,y2,y3)*(-x3+y3),(x1-y1)*(x2-y2))+2*( \
      1-2*nu)**2*(x3+y3)*np.arctan2(r2(y1,y2,y3)*(x3+y3),(x1-y1)*(x2+(-1) \
      *y2))+xLogy(-((-3)+4*nu)*(x1-y1),r1(y1,y2,y3)+x2-y2)+ \
      xLogy((5+4*nu*((-3)+2*nu))*(x1-y1),r2(y1,y2,y3)+x2-y2)+ \
      xLogy(-((-3)+4*nu)*(x2-y2),r1(y1,y2,y3)+x1-y1)+xLogy((5+ \
      4*nu*((-3)+2*nu))*(x2-y2),r2(y1,y2,y3)+x1-y1))


    J3313=lambda y1,y2,y3: \
    (1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(2*r2(y1,y2,y3)**(-1)*x3*( \
      x1-y1)*y3*(x3+y3)*((x2-y2)**2+(x3+y3)**2)**(-1)+5* \
      x2*np.arctan2((-5)*x2,x1-y1)-3*x2*np.arctan2(3*x2,x1-y1) \
      +4*nu*x2*np.arctan2(-nu*x2,x1-y1)-12*nu*x2*np.arctan2( \
      3*nu*x2,x1-y1)+8*nu**2*x2*np.arctan2(-nu**2*x2,x1+(-1) \
      *y1)-4*((-1)+nu)*(x2-y2)*np.arctan2(r1(y1,y2,y3)*(x2-y2),(x1+ \
      (-1)*y1)*(x3-y3))-8*((-1)+nu)**2*(x2-y2)* \
      np.arctan2(r2(y1,y2,y3)*(x2-y2),(x1-y1)*(x3+y3))+3*y2*np.arctan2((-3) \
      *y2,x1-y1)-5*y2*np.arctan2(5*y2,x1-y1)+12*nu*y2* \
      np.arctan2((-3)*nu*y2,x1-y1)-4*nu*y2*np.arctan2(nu*y2,x1+(-1) \
      *y1)-8*nu**2*y2*np.arctan2(nu**2*y2,x1-y1)+xLogy((-4)* \
      x3,r2(y1,y2,y3)-x1+y1)+xLogy((-4)*((-1)+nu)*(x1-y1),r1(y1,y2,y3)+x3+(-1) \
      *y3)+xLogy((-8)*((-1)+nu)**2*(x1-y1),r2(y1,y2,y3)+x3+y3)+xLogy((-1) \
      *((-3)+4*nu)*(x3-y3),r1(y1,y2,y3)+x1-y1)+xLogy((-7)*x3 \
      -5*y3+12*nu*(x3+y3)-8*nu**2*(x3+y3),r2(y1,y2,y3)+x1-y1))


    J3323=lambda y1,y2,y3: \
    (1/16)*(1-nu)**(-1)*pi**(-1)*G**(-1)*(2*r2(y1,y2,y3)**(-1)*x3*( \
      x2-y2)*y3*(x3+y3)*((x1-y1)**2+(x3+y3)**2)**(-1)+5* \
      x1*np.arctan2((-5)*x1,x2-y2)-3*x1*np.arctan2(3*x1,x2-y2) \
      +4*nu*x1*np.arctan2(-nu*x1,x2-y2)-12*nu*x1*np.arctan2( \
      3*nu*x1,x2-y2)+8*nu**2*x1*np.arctan2(-nu**2*x1,x2+(-1) \
      *y2)-4*((-1)+nu)*(x1-y1)*np.arctan2(r1(y1,y2,y3)*(x1-y1),(x2+ \
      (-1)*y2)*(x3-y3))-8*((-1)+nu)**2*(x1-y1)* \
      np.arctan2(r2(y1,y2,y3)*(x1-y1),(x2-y2)*(x3+y3))+3*y1*np.arctan2((-3) \
      *y1,x2-y2)-5*y1*np.arctan2(5*y1,x2-y2)+12*nu*y1* \
      np.arctan2((-3)*nu*y1,x2-y2)-4*nu*y1*np.arctan2(nu*y1,x2+(-1) \
      *y2)-8*nu**2*y1*np.arctan2(nu**2*y1,x2-y2)+xLogy((-4)* \
      x3,r2(y1,y2,y3)-x2+y2)+xLogy((-4)*((-1)+nu)*(x2-y2),r1(y1,y2,y3)+x3+(-1) \
      *y3)+xLogy((-8)*((-1)+nu)**2*(x2-y2),r2(y1,y2,y3)+x3+y3)+xLogy((-1) \
      *((-3)+4*nu)*(x3-y3),r1(y1,y2,y3)+x2-y2)+xLogy((-7)*x3 \
      -5*y3+12*nu*(x3+y3)-8*nu**2*(x3+y3),r2(y1,y2,y3)+x2-y2))


    IU1=lambda y1,y2,y3: \
         (Lambda*epsvkk+2*G*epsv11p)*J1123(y1,y2,y3) \
                       +2*G*epsv12p*(J1223(y1,y2,y3)+J1113(y1,y2,y3)) \
                       +2*G*epsv13p*(J1323(y1,y2,y3)+J1112(y1,y2,y3)) \
        +(Lambda*epsvkk+2*G*epsv22p)*J1213(y1,y2,y3) \
                       +2*G*epsv23p*(J1212(y1,y2,y3)+J1313(y1,y2,y3)) \
        +(Lambda*epsvkk+2*G*epsv33p)*J1312(y1,y2,y3)

    IU2=lambda y1,y2,y3: \
         (Lambda*epsvkk+2*G*epsv11p)*J2123(y1,y2,y3) \
                       +2*G*epsv12p*(J2223(y1,y2,y3)+J2113(y1,y2,y3)) \
                       +2*G*epsv13p*(J2323(y1,y2,y3)+J2112(y1,y2,y3)) \
        +(Lambda*epsvkk+2*G*epsv22p)*J2213(y1,y2,y3) \
                       +2*G*epsv23p*(J2212(y1,y2,y3)+J2313(y1,y2,y3)) \
        +(Lambda*epsvkk+2*G*epsv33p)*J2312(y1,y2,y3)

    IU3=lambda y1,y2,y3: \
         (Lambda*epsvkk+2*G*epsv11p)*J3123(y1,y2,y3) \
                       +2*G*epsv12p*(J3223(y1,y2,y3)+J3113(y1,y2,y3)) \
                       +2*G*epsv13p*(J3323(y1,y2,y3)+J3112(y1,y2,y3)) \
        +(Lambda*epsvkk+2*G*epsv22p)*J3213(y1,y2,y3) \
                       +2*G*epsv23p*(J3212(y1,y2,y3)+J3313(y1,y2,y3)) \
        +(Lambda*epsvkk+2*G*epsv33p)*J3312(y1,y2,y3)


    u1= IU1(L,T/2,q3+W)-IU1(L,-T/2,q3+W)+IU1(L,-T/2,q3)-IU1(L,T/2,q3) \
       -IU1(0,T/2,q3+W)+IU1(0,-T/2,q3+W)-IU1(0,-T/2,q3)+IU1(0,T/2,q3)
    u2= IU2(L,T/2,q3+W)-IU2(L,-T/2,q3+W)+IU2(L,-T/2,q3)-IU2(L,T/2,q3) \
       -IU2(0,T/2,q3+W)+IU2(0,-T/2,q3+W)-IU2(0,-T/2,q3)+IU2(0,T/2,q3)
    u3= IU3(L,T/2,q3+W)-IU3(L,-T/2,q3+W)+IU3(L,-T/2,q3)-IU3(L,T/2,q3) \
       -IU3(0,T/2,q3+W)+IU3(0,-T/2,q3+W)-IU3(0,-T/2,q3)+IU3(0,T/2,q3)

    # rotate displacement field to reference system of coordinates
    t1=u1*np.cos(theta)-u2*np.sin(theta)
    u2=u1*np.sin(theta)+u2*np.cos(theta)
    u1=t1

    u1 = u1.real
    u2 = u2.real
    u3 = u3.real

    return u1,u2,u3


def acoth(a):
    out = 0.5 * np.log((a + 1) / (a - 1), dtype=np.complex128)
    return out


def xLogy(x,y):
    y = x*np.log(y)
    if np.isscalar(x):
        print()
    else:
        Id0 = np.where(x==0)
        y[Id0[0]] = 0.0
    return y


if __name__ =='__main__':

    """
    Example computations
    """

    L = 0.6
    W = 0.3
    T = 0.5
    theta = 0 # deg

    G = 32
    nu = 0.25

    epsv_dict = dict(
        epsv11p = 0,    epsv12p = 0,    epsv13p = 0,
                        epsv22p = 0,    epsv23p = 0,
                                        epsv33p = 1e-6,
    )
    q1 = -0.2
    q2 = 0
    q3 = -0.1

    N = 64 * 6

    x = np.linspace(-1, 1, N, endpoint=True)
    y = np.linspace(-1, 1, N, endpoint=True)
    x1, x2 = np.meshgrid(x,y)
    x3 = np.zeros(x2.shape)

    t0 = time.time()
    u1,u2,u3 = computeDisplacementVerticalShearZone(
        x1=x1, x2=x2, x3=x3,
        q1=q1, q2=q2, q3=q3,
        L=L, W=W, T=T, theta=theta,
        **epsv_dict,
        G=1, nu=nu,
    )

    u1_, u2_, u3_ = computeDisplacementVerticalShearZone(
        x1=x1, x2=x2, x3=x3,
        q1=q1, q2=q2, q3=q3,
        L=L, W=W, T=T, theta=theta,
        **epsv_dict,
        G=32, nu=nu,
    )

    print(f"{time.time() - t0:.4} secs")

    u = np.sqrt((u1**2) + (u2**2) + (u3**2))

    fig, ax = plt.subplots(figsize=(10, 8), nrows=2, ncols=2, sharex='row', sharey='col')

    cmap=cm.vik
    pu1 = ax[0, 0].pcolormesh(x1, x2, u1, cmap=cmap, norm=colors.CenteredNorm(vcenter=0))
    pu2 = ax[0, 1].pcolormesh(x1, x2, u2, cmap=cmap, norm=colors.CenteredNorm(vcenter=0))
    pu3 = ax[1, 0].pcolormesh(x1, x2, u3, cmap=cmap, norm=colors.CenteredNorm(vcenter=0))
    pu = ax[1, 1].pcolormesh(x1, x2, u)

    ax[0, 0].set_title('Displacement u1 (m)')
    ax[0, 1].set_title('Displacement u2 (m)')
    ax[1, 0].set_title('Displacement u3 (m)')
    ax[1, 1].set_title('Total Displacement u (m)')

    ax[0, 0].set_ylabel('x2 (m)')
    ax[1, 0].set_ylabel('x2 (m)')
    ax[1, 0].set_xlabel('x1 (m)')
    ax[1, 1].set_xlabel('x1 (m)')

    for axis in ax.flatten():
        axis.set_aspect('equal')
        axis.ticklabel_format(axis='both', style='scientific', useMathText=True, scilimits=(0, 0))
        axis.grid(alpha=0.2)
        axis.plot(q1, q2, '^', zorder=100)

        c1 = q1 + (L/2)
        c2 = q2
        print(f'Start {q3=} |<---- c3={(q3 + (W/2))=} ---->| {q3 + W} End')
        axis.plot(c1, c2, '^', zorder=100)

        axis.plot([q1, q1+L], [q2+T/2, q2+T/2], '--', color='gray')
        axis.plot([q1, q1+L], [q2-T/2, q2-T/2], '--', color='gray')
        axis.plot([q1, q1], [q2-T/2, q2+T/2], '-', color='gray')
        axis.plot([q1+L, q1+L], [q2-T/2, q2+T/2], '-', color='gray')

    plotted_nonzero_strain_comp = ""
    for k, v in epsv_dict.items():
        if v != 0:
            plotted_nonzero_strain_comp += f"{k}:{v:.2e}_"

    fig.suptitle(plotted_nonzero_strain_comp)

    fig.colorbar(pu1, ax=ax[0, 0])
    fig.colorbar(pu2, ax=ax[0, 1])
    fig.colorbar(pu3, ax=ax[1, 0])
    fig.colorbar(pu, ax=ax[1, 1])

    plt.show()