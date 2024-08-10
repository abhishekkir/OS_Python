#MIT License

#Copyright (c) 2024 Abhishek Kumar

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np
import numpy.linalg as LA
from scipy.linalg import eig

class stability_analysis:
    def __init__(self, x1, x2, x3, x4):
          self.Re = x1
          self.N = x2
          self.alpha = x3
          self.beta = x4
          
          
    def cheb(self):
        #First we generate the Chebyshev grid x
        j = np.arange(0, self.N+1) #N is the order and N + 1 is the no. of points.
        nodes =  (j*np.pi)/self.N
        x = np.cos(nodes)
        
        #Now we will set the differentiation matrix D.
        c = np.ones(self.N+1)
        c[0] = 2.0
        c[self.N] = 2.0
        c = c*(-1.0)**j
        c.shape = (self.N+1, 1)
        x.shape = (self.N+1, 1)
        X = np.tile(x, (1, self.N+1) )
        dX = (X - X.T) + np.eye(self.N+1)
        D  = np.dot(c, 1.0/c.T) / (dX)
        D  = D - np.diag(D.sum(axis=1))
        return(D, x)
    
    def eig_value(self):
        D, x = self.cheb()
        
        #Construct D2
        D2 = np.dot(D,D)
        D2 = D2[1:-1, 1:-1] #Implementation of boundary conditions


        #Construct discrete biharmonic operator D4
        S = np.diag(np.insert(1/(1-x[1:-1]**2), [0, self.N-1], [0, 0]))
        D4_first_term = np.dot(np.diag(1 - x[:,0]**2), LA.matrix_power(D, 4)) - (8 * np.dot(np.diag(x[:,0]),  LA.matrix_power(D, 3)) ) - (12*LA.matrix_power(D, 2))
        D4 = np.dot(D4_first_term, S)
        D4 = D4[1:-1,1:-1] #Implementation of boundary conditions

        U = np.diag(1 - x[1:-1][:,0]**2)
        U_2 = - 2
        zi = 0+1.j
        k2 = (self.alpha**2)+(self.beta**2)
        k4 = k2**2
        I = np.eye(self.N - 1)
        A = (((D4 - (2*k2*D2) + k4*I)*zi)/self.Re) + (self.alpha * np.dot(U, D2-(k2*I))) - (self.alpha*U_2*I)
        B = D2 - (k2*I)

        eigval,eigvec = eig(A,B)
        return(eigval,eigvec)
    
    def get_most_unstable_eigenvalue(self):
        eigval,eigvec = self.eig_value()
        Real_eig_val = np.real(eigval)
        Im_eig_val = np.imag(eigval)
        max_index = np.argmax(Im_eig_val)
        return(Real_eig_val[max_index],Im_eig_val[max_index])

    def get_u_for_2D(self, v):
        D, y = self.cheb()
        zi = 0+1.j
        v_hat_y = np.dot(D,v)
        u = (zi/self.alpha)*v_hat_y
        return(u)

#get_u_for_2D(np.linspace(0,1,100),1)
#lsa_OS = stability_analysis(1e4, 0 , 1, 0)
#Cheb_points = np.arange(20,55,5)
#for i in range(0,len(Cheb_points)):
#    lsa_OS.N = Cheb_points[i]
#    temp = lsa_OS.get_most_unstable_eigenvalue()
#    print( Cheb_points[i],temp[0])
