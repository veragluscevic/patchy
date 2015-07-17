from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fact

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
import glob
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times','Palatino']})
rc('text', usetex=True)
mpl.rcParams['xtick.major.size']=8
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['xtick.labelsize']=18
mpl.rcParams['ytick.labelsize']=18
import os,os.path,shutil

#from sympy import N
#from sympy.physics.wigner import wigner_3j
import pywigxjpf as wig


lmax = 5000
wig.wig_table_init(2*lmax,9)
wig.wig_temp_init(2*lmax)

# Free memory space
#wig.wig_temp_free()
#wig.wig_table_free()

Tcmb = 2.7255 * 1e6 # CMB temperature in uK
spectra = np.loadtxt('scalCls.dat')
ls = spectra[:lmax-1,0]
tt = spectra[:lmax-1,1] * (2.*np.pi) / ls / (ls + 1) #/ Tcmb
NET = 30 # Planck noise in uK(sec)**0.5
tobs = 2.5 # observation time in years
fsky = 0.75

fwhm = 5. #Planck resolution in arcmin
sigma_b = 0.00741 * fwhm / 60.
Window = 1./np.exp(ls**2*sigma_b**2/2.)

Noise = 4.*np.pi*fsky * NET**2 / (tobs * 365*24*3600.)
tt_tilde = tt #* Window**2
tt_map = tt_tilde + Noise / Window**2

LS = np.array([2,47,101,200,301,400,501,944,1000,1101,1200,1301,1400,1500,1750,2000])

def stirling(num):
    return num*np.log(num) - num


def get_Fks(J, filename='Fks_10000.txt'):
    Ftable = np.loadtxt(filename)
    Ftable = np.atleast_1d(Ftable)
    return Ftable[J]

def tabulate_Fks(Jmax=10000, filename='Fks_10000.txt'):
    #filename='Fks_{}.txt'.format(Jmax)
    if(os.path.exists(filename)):
        fin = open(filename, 'w')
        fin.close()
    
    for J in np.arange(Jmax):
        print J
        if J==0:
            Fres = 1
        else:
            Fres = np.sqrt(1-1./(2*J)) * get_Fks(J-1, filename)
            
        fout = open(filename, 'a')
        fout.write('{:.20f}\n'.format(Fres))
        fout.close()

#def Fks(J):
#    """this uses Kendrick Smith's recursion formula"""
#    if J==0:
#        return 1.
#    res = np.sqrt(1-1./(2*J)) * Fks(J-1)
#    return res

    
def w3j_000_KS(L, l, lp):
    """this uses Kendrick Smith's recursion formula"""

    J = L + l + lp    
    if (J % 2 == 1) or (l + lp < L) or (np.abs(l - lp) > L):
        return 0.

    res = (-1)**(J/2) * (get_Fks(J/2 - L) * get_Fks(J/2 - l) * get_Fks(J/2 - lp)) / (get_Fks(J/2) * (J + 1.)**0.5)
    
    return res

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

    #lnfact1 = stirling(J-2.*L) + stirling(J-2.*l) + stirling(J-2.*lp) - stirling(J+1.)
    #lnfact2 = stirling(J/2.) - stirling(J/2.-L) - stirling(J/2.-l) - stirling(J/2.-lp)
    #res = (-1.)**(J/2.) * np.exp(0.5*lnfact1 + lnfact2)
    fact1 = fact(J-2*L) * fact(J-2*l) * fact(J-2*lp) / fact(J+1)
    fact2 = fact(J/2) / (fact(J/2-L) * fact(J/2-l) * fact(J/2-lp))
    res = (-1)**(J/2) * fact1**0.5 * fact2
    return res

def w3j_factor(L, l, lp):
    """This computes the wigner-3j symbol for m's all zeros
    """
    #w3j = N(wigner_3j(L, l, lp, 0, 0, 0))
    #w3j = w3j_000(L, l, lp)
    w3j = val3j = wig.wig3jj([2*L, 2*l, 2*lp, 0, 0, 0])
    res = w3j**2 * (2.*l + 1.) * (2.*lp + 1.) / (4.*np.pi)

    return res

def powerTT_factor(L, l, lp, fullsky=True):
    """This computes the power-spectrum factor in
    the variance of C_L^\tau\tau, for TT estimator
    """
    l_ind = l - 2
    lp_ind = lp - 2
    num = (tt_tilde[l_ind] + tt_tilde[lp_ind])**2.
    denom = tt_map[l_ind] * tt_map[lp_ind]
    if l==lp:
        if fullsky:
            denom += (-1)**L*tt_map[l_ind]

    res = num / denom
    return res

def sigma_L_old(L):
    """This computes the variance of the tau estimator from TT for a single L
    """

    w3js = np.load('/data/verag/wig3j/J{}_2000.npy'.format(L))
    sum = 0.
    for i,l in enumerate(ls):
        for j,lp in enumerate(ls):
            w3jfactor = w3js[i,j]**2 * (2.*l + 1.) * (2.*lp + 1.) / (4.*np.pi)
            #sum += w3j_factor(L, l, lp) * powerTT_factor(L, l, lp)
            sum += w3jfactor * powerTT_factor(L, l, lp)
    res = 1./sum

    return res

def sigma_L(L):
    """This computes the variance of the tau estimator from TT for a single L
    """

    Flist = np.loadtxt('Fks_10000.txt')
    
    sum = 0.
    for i,l in enumerate(ls):
        for j,lp in enumerate(ls):
            J = L + l + lp    
            if not((J % 2 == 1) or (l + lp < L) or (np.abs(l - lp) > L)):
                w3j = (-1)**(J/2) * (Flist[J/2 - L] * Flist[J/2 - l] * Flist[J/2 - lp]) / (Flist[J/2] * (J + 1.)**0.5)
                w3jfactor = w3j**2 * (2.*l + 1.) * (2.*lp + 1.) / (4.*np.pi)
                sum += w3jfactor * powerTT_factor(L, l, lp)
    res = 1./sum

    return res

def sigmas_old(Lmin=2,Lmax=2000,NLs=40,
           outfile='sigmas_NET30_2.5yr.txt',
           flatsky=False):
    """This calls sigma_L for an array of L's
    """
    #Lstep = int((Lmax - Lmin) / float(NLs))
    #Ls = np.arange(Lmin, Lmax, Lstep)
    
    #s = np.zeros((2,len(Ls)))
    s = np.zeros((2,len(LS)))
    #s[0] = Ls
    s[0] = LS
    #for i,L in enumerate(Ls):
    for i,L in enumerate(LS):
        print 'for L={}'.format(L)
        if flatsky:
            s[1,i] = sigma_L_flatsky(L)
        else:
            s[1,i] = sigma_L(L)
        print s[1,i]

    np.savetxt(outfile,s)
    return s

def sigmas(Lmin=2,Lmax=2000,NLs=40,
           outfile='sigmas_NET30_2.5yr_KS.txt',
           flatsky=False):
    """This calls sigma_L for an array of L's
    """
    Lstep = int((Lmax - Lmin) / float(NLs))
    Ls = np.arange(Lmin, Lmax, Lstep)
    
    s = np.zeros((2,len(Ls)))
    s[0] = Ls

    for i,L in enumerate(Ls):
        print 'for L={}'.format(L)
        s[1,i] = sigma_L(L)
        print s[1,i]

    np.savetxt(outfile,s)
    return s

        
def sigma_L_flatsky_old(L):
    """This computes the variance of the tau estimator from TT for a single L
    """

    sum = 0.
    for l in ls:
        for lp in ls:
            if L > l+lp or L < np.abs(l-lp):
                sum += 0.
            else:
                summand = (2.*np.pi)**2 * powerTT_factor(L, l, lp, fullsky=False)
                #summand = 1./(2.*np.pi)**2 * l*lp* powerTT_factor(L, l, lp, fullsky=False)
                if l==lp:
                    summand *= 2.
                sum += summand
    res = 1./sum

    return res


def plot_model_sigma(modelnums=np.array([0,1]),
                     sfile='sigmas_NET30_2.5yr_KS.txt'):

    
    
    plt.figure()
    s = np.loadtxt(sfile)
    plt.semilogy(s[0,:], s[1,:]*(2./(2.*s[0,:]+1)),'--',lw=3,color='gray',label='Planck TT upper limit')
    for mnum in modelnums: 
        mfile = 'model{}.txt'.format(mnum)
        m0 = np.loadtxt(mfile)
        label = ''
        color = 'blue'
        if mnum==0:
            label = 'fiducial model'
            color = 'red'
        if mnum==1:
                label = 'hi model'
                color = 'Maroon'
        
        plt.semilogy(m0[0,:], m0[1,:], lw=2,color=color, label=label)
    plt.xlim(xmax=2000)
    plt.legend(loc='lower left',frameon=False,fontsize=20)
    plt.title(r'$C_\ell^{\tau\tau}$',fontsize=22)
    plt.savefig('sensitivity_Planck_all.png')

    
    
