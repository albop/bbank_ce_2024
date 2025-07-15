σ = 0.05; γ = 40

from math import exp
import numpy as np
from numpy.random import normal
from matplotlib import pyplot as plt

U = lambda x: (x**(-γ))/(-γ)
C = lambda e: U(exp(e))

def E_ϵ(f, N=100):

    gen = (f(normal()*σ) for e in range(N))
    return sum(gen)/N

NVec = [1000, 5000, 10000, 15000, 20000]
vals = [E_ϵ(C, N=i) for i in NVec]

plt.plot(NVec, vals)



def stdev(f, N=100, K=100): 
    gen = (E_ε(f,N=N) for k in range(K))
    return np.std([*gen])

sdvals = [stdev(C, N=n, K=1000) for n in NVec] 

plt.plot(NVec, sdvals)


from numpy import polynomial
from math import sqrt,pi

x, w = polynomial.hermite_e.hermegauss(4)
x = x*σ # renormalize nodes
s = sum( w_*U(exp(x_)) for (x_,w_) in zip(x,w))/sqrt(pi)/sqrt(2)
print(s)


# should be an exercise