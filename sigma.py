from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fact

#from sympy import N
#from sympy.physics.wigner import wigner_3j

Tcmb = 2.7255 * 1e6 # CMB temperature in uK
lmax = 2000
spectra = np.loadtxt('scalCls.dat')
ls = spectra[:lmax-1,0]
tt = spectra[:lmax-1,1] * (2.*np.pi) / ls / (ls + 1) #/ Tcmb
NET = 30 # Planck noise in uK(sec)**0.5
tobs = 2 # observation time in years
fsky = 0.75

fwhm = 5. #Planck resolution in arcmin
sigma_b = 0.00741 * fwhm / 60.
Window = 1./np.exp(ls**2*sigma_b**2/2.)

Noise = 4.*np.pi*fsky * NET**2 / (tobs * 365*24*3600.)
tt_tilde = tt #* Window**2
tt_map = tt_tilde + Noise / Window**2


def w3j_000(L, l, lp):

    J = L + l + lp
    if J % 2 == 1:
        #print 'odd J\n'
        return 0.
    if l + lp < L:
        #print 'l + lp < L'
        return 0.
    if np.abs(l - lp) > L:
        #print 'abs(l - lp) > L'
        return 0.

    fact1 = fact(J-2*L) * fact(J-2*l) * fact(J-2*lp) / fact(J+1)
    fact2 = fact(J/2) / (fact(J/2-L) * fact(J/2-l) * fact(J/2-lp))
    res = (-1)**(J/2) * fact1**0.5 * fact2
    return res

def w3j_factor(L, l, lp):
    """This computes the wigner-3j symbol for m's all zeros
    """
    #w3j = N(wigner_3j(L, l, lp, 0, 0, 0))
    w3j = w3j_000(L, l, lp)
    res = w3j**2 * (2.*l + 1.) * (2.*lp + 1.) / (4.*np.pi)

    return res

def powerTT_factor(L, l, lp):
    """This computes the power-spectrum factor in
    the variance of C_L^\tau\tau, for TT estimator
    """
    l_ind = l - 2
    lp_ind = lp - 2
    num = (tt_tilde[l_ind] + tt_tilde[lp_ind])**2.
    denom = tt_map[l_ind] * tt_map[lp_ind]
    if l==lp:
        denom += (-1)**L*tt_map[l_ind]

    res = num / denom
    return res

def sigma_L(L):
    """This computes the variance of the tau estimator from TT for a single L
    """

    sum = 0.
    for l in ls:
        #print l
        for lp in ls:
           sum += w3j_factor(L, l, lp) * powerTT_factor(L, l, lp)
    res = 1./sum

    return res

def sigmas(Lmin=2,Lmax=2000,NLs=40,outfile='sigmas_test.txt'):
    """This calls sigma_L for an array of L's
    """
    Lstep = int((Lmax - Lmin) / float(NLs))
    Ls = np.arange(Lmin, Lmax, Lstep)
    
    s = np.zeros((2,len(Ls)))
    s[0] = Ls
    for i,L in enumerate(Ls):
        print 'for L={}'.format(L)
        s[1,i] = sigma_L(L)

    np.savetxt(outfile,s)
    return s
        
