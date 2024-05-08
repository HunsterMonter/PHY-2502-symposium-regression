from calendar import isleap
from datetime import date, datetime, timedelta
from functools import partial
from math import modf
from scipy.odr import ODR, Model, RealData
from scipy.optimize import curve_fit, root_scalar
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)


def jour_to_date (annee, jour):
    leap = isleap (annee)

    jours_mois = [31, 28+leap, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    mois = 1

    for mo in jours_mois:
        if jour <= mo:
            break

        else:
            jour -= mo
            mois += 1

    return datetime (annee, mois, jour, 0)


def dec_to_date (annee_dec):
    dec, annee = modf (annee_dec)
    annee = int (annee)
    leap = isleap (annee)

    jour_dec = dec * (365+leap)
    jour = int (np.rint (jour_dec))

    date = jour_to_date (annee, jour)

    return date


# Données brutes
annee_prev = np.array ([1690, 1712, 1715, 1749.25, 1750.25, 1753, 1756, 1764, 1769, 1771]) + 0.5
annee = np.array ([1781, 1781, 1782, 1782, 1782, 1783, 1783, 1783, 1784, 1784, 1785, 1785, 1785, 1786, 1786, 1787, 1788, 1788, 1789, 1789, 1790, 1790, 1791, 1791, 1792, 1792, 1793, 1794, 1794, 1795, 1795, 1796, 1796, 1797, 1797, 1798, 1799, 1799, 1800, 1801, 1802, 1802, 1803, 1804, 1805, 1806, 1806, 1807, 1807, 1808, 1809, 1810, 1811, 1811, 1812, 1812, 1813, 1813, 1814, 1814, 1815, 1815, 1816, 1816, 1817, 1818, 1819, 1820, 1821]) + 0.5
annee_next = np.array ([1833, 1833, 1833, 1833, 1833, 1833, 1834, 1834, 1834, 1834, 1834, 1835, 1835, 1835, 1835, 1835, 1835, 1836, 1836, 1836, 1836, 1836, 1836])
jour_next = np.array ([215, 235, 265, 279, 295, 323, 229, 258, 283, 312, 336, 215, 232, 255, 293, 326, 337, 203, 236, 256, 285, 307, 327])
sigma_annee = 0.25

t = np.array ([0.0076, 0.0100, 0.0123, 0.0180, 0.0205, 0.0227, 0.0288, 0.0313, 0.0335, 0.0393, 0.0421, 0.0442, 0.0508, 0.0530, 0.0554, 0.0640, 0.0752, 0.0769, 0.0862, 0.0886, 0.0975, 0.1063, 0.1088, 0.1174, 0.1201, 0.1287, 0.1315, 0.1438, 0.1513, 0.1541, 0.1626, 0.1654, 0.1740, 0.1767, 0.1852, 0.1877, 0.1989, 0.2068, 0.2098, 0.2206, 0.2290, 0.2314, 0.2433, 0.2523, 0.2626, 0.2703, 0.2725, 0.2800, 0.2835, 0.2922, 0.3015, 0.3106, 0.3179, 0.3198, 0.3268, 0.3287, 0.3354, 0.3373, 0.3444, 0.3458, 0.3519, 0.3539, 0.3600, 0.3618, 0.3697, 0.3781, 0.3878, 0.3926, 0.3996])
sigma_t = 0.00005

err_prev = -np.pi/(200*10000) * np.array ([132.7, 202.7, 135.7, -227.7, -206.4, -186.6, -203.5, -151.5, -99.1, -46.9])
err = -np.pi/(200*10000) * np.array ([-17.2, -10.3, -4.7, -10.3, -13.0, -1.0, 2.7, -3.8, 0.4, -9.0, -7.7, -0.5, 2.3, 2.1, 5.2, -10.8, 11.7, 2.9, 32.2, 24.0, -4.6, 4.0, 12.0, 12.7, 4.9, 4.2, 10.8, 8.3, 8.2, 3.3, 16.4, -1.9, 12.8, -5.9, -5.3, -20.3, -12.8, 6.3, -12.5, -19.2, -4.9, -15.1, -2.3, -14.5, -2.7, 8.5, -10.6, 11.6, -2.1, -13.3, -4.5, 3.8, -9.4, -5.9, 4.7, -2.4, 17.8, 1.6, -11.9, 7.7, 8.0, 8.7, 9.7, 2.8, -2.1, 9.4, -0.6, 0.3, -4.2])
err_next = np.pi/(180*3600) * (np.array ([31.20, 31.53, 32.03, 32.27, 32.53, 33.00, 37.47, 37.97, 38.41, 38.92, 39.34, 43.88, 44.21, 44.66, 45.40, 46.05, 46.26, 51.06, 51.73, 52.14, 52.74, 53.19, 53.60]) + 0.00337 * np.array ([284, 286, 290, 291, 293, 296, 337, 335, 333, 331, 329, 248, 245, 241, 234, 228, 226, 210, 207, 205, 203, 201, 199]))

# Erreur en rad
err = np.concatenate ((err_prev[:], err, err_next))
#err = np.concatenate ((err_prev[3:], err))


# Calcule l'année décimale des données
leap = np.array ([isleap (a) for a in annee_next])
annee_dec_next = np.where (leap, annee_next + jour_next/366, annee_next + jour_next/365)
#annee_dec = np.concatenate ((annee_prev[3:], annee_dec, annee_dec_next))
#annee_dec = np.concatenate ((annee_prev[3:], annee_dec))

# Affiche le nombre de points qui sont mal repésentés par la régression
#print (np.sum ([abs (root_scalar (f_i, x0=xi).root - a) > 0.5 for xi, f_i, a in zip(x0, f_v, annee)]))
# Affiche les paramètres de la régression
#print (output.b)

# Date de référence
epoch = datetime (1800, 1, 1, 0)
# Calcule le nombre de jours depuis le 1er janvier 1800
#jours_ecoules = np.array([(dec_to_date (an) - epoch).days for an in annee_dec], int)
# Temps en 2*pi/période
#temps = jours_ecoules/365.25 * 2*np.pi/84.02


temps = np.array([-0.8915, -0.7579, -0.7267, -0.2837, -0.2817, -0.2525, -0.2276, -0.1630, -0.1201, -0.0957, 0.0076, 0.0100, 0.0123, 0.0180, 0.0205, 0.0227, 0.0288, 0.0313, 0.0335, 0.0393, 0.0421, 0.0442, 0.0508, 0.0530, 0.0554, 0.0640, 0.0752, 0.0769, 0.0862, 0.0886, 0.0975, 0.1063, 0.1088, 0.1174, 0.1201, 0.1287, 0.1315, 0.1438, 0.1513, 0.1541, 0.1626, 0.1654, 0.1740, 0.1767, 0.1852, 0.1877, 0.1989, 0.2068, 0.2098, 0.2206, 0.2290, 0.2314, 0.2433, 0.2523, 0.2626, 0.2703, 0.2725, 0.2800, 0.2835, 0.2922, 0.3015, 0.3106, 0.3179, 0.3198, 0.3268, 0.3287, 0.3354, 0.3373, 0.3444, 0.3458, 0.3519, 0.3539, 0.3600, 0.3618, 0.3697, 0.3781, 0.3878, 0.3926, 0.3996])
cosphi = 0.0001 * np.array([-1888, 9856, 9988, -9089, -9130, -9809, -9999, -8470, -5829, -4058, 3087, 3248, 3387, 3790, 3954, 4077, 4477, 4638, 4794, 5136, 5293, 5443, 5798, 5920, 6040, 6512, 7069, 7155, 7579, 7685, 8054, 8481, 8479, 8771, 8857, 9111, 9182, 8456, 9627, 9674, 9804, 9838, 9926, 9945, 9990, 9996, 9989, 9948, 9924, 9804, 9670, 9625, 9393, 9104, 8768, 8475, 8373, 8053, 7943, 7460, 6936, 6369, 5904, 5763, 5284, 5121, 4619, 4456, 3946, 3769, 3237, 3075, 2525, 2335, 1598, 843, 91, -640, -1390])
e = 0.0466108
temps *= 100 / (1 + 2*e*cosphi) # Temps en années depuis 1781
temps -= 19 # Temps en années depuis 1800
temps = temps[:]
temps = np.concatenate((temps, np.array([(dec_to_date (an) - epoch).days for an in annee_dec_next])/365.25))
temps *= 2*np.pi / 84.02 # Temps en omega_3 depuis 1800


tempsV = np.array([1690.98, 1712.25, 1715.23, 1747.7, 1754.7, 1761.7, 1768.7, 1775.7, 1782.7, 1789.7, 1796.7, 1803.7, 1810.7, 1817.7, 1824.7, 1831.7, 1838.7, 1845.7])
tempsV = (tempsV - 1800) * 2*np.pi / 84.02
errV = np.pi / (180 * 3600) * np.array([-63.1, -59.9, -64.6, 34.8, 32.8, 24.7, 10.0, -3.7, -17.4, -28.6, -29.8, -33.6, -35.3, -32.3, -24.5, 3.4, 50.0, 110.5])

# Paramètres orbitaux d'Uranus

#a3 = 1
#e3 = 0.047
#t_03 = 6.182
#phi_03 = 2.969
#t_03 = 3.255
#phi_03 = 2.935
#t_03 = 6.149
#phi_03 = 2.683

#t_03 = 0
#phi_03 = 2.818
#phi_03 = 3.077

# Nouvelles variables canoniques d'Uranus

#a_13 = 0.8
#a_23 = 1/a_13
#b_13 = 1.5
#b_23 = -0.4

# La fonction insane à régresser wow crazyyyy
def phi1_13(t, m4, a4, e4, t_04, phi_04, h1, h2, h3, h4):
    a3 = 1 + h1
    e3 = 0.047 + h2
    t_03 = 6.155 + h3
    phi_03 = 2.939 + h4

    a_13 = 1 / a3**0.5
    a_23 = np.sqrt(a3 * (1-e3**2))
    b_13 = t_03 / a3**0.5 + np.pi/2 * a3
    b_23 = phi_03
    
    a_14 = 1 / a4**0.5
    a_24 = np.sqrt(a4 * (1-e4**2))
    b_14 = t_04 / a4**0.5 + np.pi/2 * a4
    b_24 = phi_04
    
    A = a_13**2 * a_14**2 / (a_13**4 + a_14**4)**0.5
    pdvA = 2 * a_13 * a_14**6 / (a_13**4 + a_14**4)**1.5
    B = 2 * a_13**2 * a_14**2 / (a_13**4 + a_14**4)
    pdvB = -4 * a_13 * a_14**2 * (a_13**4 - a_14**4) / (a_13**4 + a_14**4)**2
    
    B3 = a_13**3*t - a_13**2*b_13
    B4 = a_14**3*t - a_14**2*b_14
    A3 = B3 + b_23 + np.pi/2
    A4 = B4 + b_24 + np.pi/2
    
    omega = a_13**3 - a_14**3
    delta = b_23 - b_24 - a_13**2*b_13 + a_14**2*b_14
    c = a_13**2 * a_14**2 * a_23 * a_24

    c0 = A*(1 + 3*B**2/16)
    c1 = A*(B/2 + 15*B**3/64)
    c2 = A*(3*B**2/16)
    c3 = A*(5*B**3/64)

    cn = [c0, c1, c2, c3]

    pdvC0 = A*(3*B/8)
    pdvC1 = A*(1/2 + 45*B**2/64)
    pdvC2 = A*(3*B/8)
    pdvC3 = A*(15*B**2/64)

    pdvCn = [pdvC0, pdvC1, pdvC2, pdvC3]

    def sum_1(t):
        s = 0
        for n in range(1, len(cn)):
            s += cn[n] * np.cos(n*(omega*t + delta))

        return s

    def sum_2(t):
        s = 0
        for n in range(1, len(cn)):
            s += cn[n]/n * np.sin(n*(omega*t + delta))

        return s

    def sum_3(t):
        s = 0
        for n in range(1, len(cn)):
            s += (cn[n]*pdvA/A + pdvCn[n]*pdvB)/n * np.sin(n*(omega*t + delta))

        return s
    
    d1 = -A**3 * (1/a_13**2 - 3*B/(4*a_14**2))
    d2 = a_13**3*a_14**2*a_24 * (3*a_13*a_23 - 1) - 2*A**3 * (1/(2*a_14**2) + 3*B/(4*a_13**2))
    d3 = -1/2*a_13**4*a_14*a_23 * (a_14*a_24 - 1) + a_13**2*A**3/a_14**2 * (3/(2*a_13**2) - 3*B/(4*a_14**2))
    d4 = -1/2*a_13**4*a_14*a_23 * (3*a_14*a_24 - 1) + a_13**2*A**3/a_14**2 * (1/(2*a_13**2) + 3*B/(4*a_14**2))
    d5 = A**3/a_13**2 * ((3/A*pdvA - 2/a_13)*(1/a_13**2 - 3*B/(4*a_14**2)) - (2/a_13**3 + 3*pdvB/(4*a_14**2)))
    d6 = -1/2 * a_14**2*a_24 * (6*a_13*a_23 - 1) + A**3/a_13**2 * ((3/A*pdvA - 2/a_13)*(1/(2*a_14**2) + 3*B/(4*a_13**2)) - 3/(4*a_13**2)*(2*B/a_13 - pdvB))
    d7 = -1/2 * a_14**2*a_24 * (2*a_13*a_23 - 1) + A**3/a_13**2 * ((3/A*pdvA - 2/a_13)*(3/(2*a_14**2) - 3*B/(4*a_13**2)) + 3/(4*a_13**2)*(2*B/a_13 - pdvB))
    d8 = A**3/a_13**2 * ((3/A*pdvA - 2/a_13)*3*B/(8*a_14**2) + 3/(8*a_14**2)*pdvB)
    d9 = A**3/a_13**2 * (1/a_13**2 - 3*B/(4*a_14**2))
    d10 = -a_13*a_14**2*a_24 * (3*a_13*a_23 - 1) + 2*A**3/a_13**2 * (1/(2*a_14**2) + 3*B/(4*a_13**2))
    d11 = A**3/a_14**2 * (3/A*pdvA * (1/a_14**2 - 3*B/(4*a_13**2)) + 3/(4*a_13**2) * (2*B/a_13 - pdvB))
    d12 = a_13*a_14*a_23 * (a_14*a_24 - 1) + A**3/a_14**2 * (-3/A*pdvA * (3/(2*a_13**2) - 3*B/(4*a_14**2)) + (3/a_13**3 + 3/(4*a_14**2)*pdvB))
    d13 = a_13*a_14*a_23 * (3*a_14*a_24 - 1) + A**3/a_14**2 * (-3/A*pdvA * (1/(2*a_13**2) + 3*B/(4*a_14**2)) + (1/a_13**3 - 3/(4*a_14**2)*pdvB))
    d14 = A**3/a_14**2 * (-3/A*pdvA * 3*B/(8*a_13**2) + 3/(8*a_13**2)*(2*B/a_13 - pdvB))
    d15 = 1/2*a_13**2*a_14*a_23 * (a_14*a_24 - 1) - A**3/a_14**2 * (3/(2*a_13**2) - 3*B/(4*a_14**2))
    d16 = 1/2*a_13**2*a_14*a_23 * (3*a_14*a_24 - 1) - A**3/a_14**2 * (1/(2*a_13**2) + 3*B/(4*a_14**2))
    d17 = -1/2*a_13*a_14**2*a_24 * (3*a_13*a_23 - 1) + A**3/a_13**2 * (1/(2*a_14**2) + 3*B/(4*a_13**2))
    d18 = -1/2*a_13*a_14**2*a_24 * (a_13*a_23 - 1) + A**3/a_13**2 * (3/(2*a_14**2) - 3*B/(4*a_13**2))
    
    a1_13_func = lambda t: m4 * ( a_13**2/omega * ( c*np.cos(omega*t + delta) - sum_1(t) )
                       - e3 * (d1/a_13**3 * np.sin(a_13**3*t - a_13**2*b_13) + d2/(omega+a_13**3)*np.sin((omega+a_13**3)*t + delta - a_13**2*b_13) - 9*A**3*B/(8*a_14**2) * (1/(2*omega+a_13**3)*np.sin((2*omega+a_13**3)*t + 2*delta - a_13**2*b_13) + 1/(2*omega-a_13**3)*np.sin((2*omega-a_13**3)*t + 2*delta + a_13**2*b_13)))
                       - e4 * (d3/(omega+a_14**3)*np.sin((omega+a_14**3)*t + delta - a_14**2*b_14) + d4/(omega-a_14**3)*np.sin((omega-a_14**3)*t + delta + a_14**2*b_14) + 3*A**3*B/(4*a_14**2) * (3/(2*omega+a_14**3)*np.sin((2*omega+a_14**3)*t + 2*delta - a_14**2*b_14) + 1/(2*omega-a_14**3)*np.sin((2*omega-a_14**3)*t + 2*delta + a_14**2*b_14)))
                      )
    a1_13 = a1_13_func(t) - a1_13_func(0)

    b1_13_func = lambda t: m4 * ( 2*c/(a_13*omega) * np.sin(omega*t + delta) - (cn[0]*pdvA/A + pdvCn[0]*pdvB)*t - sum_3(t)/omega - 3*a_13**2/omega**2 * (c*np.sin(omega*t + delta) - sum_2(t)) + (3*a_13**2*t - 2*a_13*b_13)/omega * (c*np.cos(omega*t + delta) - sum_1(t))
                       + e3 * ( -d5/a_13**3*np.cos(a_13**3*t - a_13**2*b_13) - d6/(omega+a_13**3)*np.cos((omega+a_13**3)*t + delta - a_13**2*b_13) - d7/(omega-a_13**3)*np.cos((omega-a_13**3)*t + delta + a_13**2*b_13) - d8/(2*omega+a_13**3)*np.cos((2*omega+a_13**3)*t + 2*delta - a_13**2*b_13) - 3*d8/(2*omega-a_13**3)*np.cos((2*omega-a_13**3)*t + 2*delta + a_13**2*b_13)
                               + (3*a_13**2*t - 2*a_13*b_13) * ( d9/a_13**3*np.sin(a_13**3*t - a_13**2*b_13) + d10/(omega+a_13**3)*np.sin((omega+a_13**3)*t + delta - a_13**2*b_13) + 9*A**3*B/(8*a_13**2*a_14**2) * (1/(2*omega+a_13**3)*np.sin((2*omega+a_13**3)*t + 2*delta - a_13**2*b_13) + 1/(2*omega-a_13**3)*np.sin((2*omega-a_13**3)*t + 2*delta + a_13**2*b_13)) )
                               + 3*a_13**2 * ( d9/a_13**6*np.cos(a_13**3*t - a_13**2*b_13) + d10/(omega+a_13**3)**2*np.cos((omega+a_13**3)*t + delta - a_13**2*b_13) + 9*A**3*B/(8*a_13**2*a_14**2) * (1/(2*omega+a_13**3)**2*np.cos((2*omega+a_13**3)*t + 2*delta - a_13**2*b_13) + 1/(2*omega-a_13**3)**2*np.cos((2*omega-a_13**3)*t + 2*delta + a_13**2*b_13)) )
                              )
                       + e4 * ( -d11/a_14**3*np.cos(a_14**3*t - a_14**2*b_14) - d12/(omega+a_14**3)*np.cos((omega+a_14**3)*t + delta - a_14**2*b_14) - d13/(omega-a_14**3)*np.cos((omega-a_14**3)*t + delta + a_14**2*b_14) - 3*d14/(2*omega+a_14**3)*np.cos((2*omega+a_14**3)*t + 2*delta - a_14**2*b_14) - d14/(2*omega-a_14**3)*np.cos((2*omega-a_14**3)*t + 2*delta + a_14**2*b_14)
                               + (3*a_13**2*t - 2*a_13*b_13) * ( d15/(omega+a_14**3)*np.sin((omega+a_14**3)*t + delta - a_14**2*b_14) + d16/(omega-a_14**3)*np.sin((omega-a_14**3)*t + delta + a_14**2*b_14) - 3*A**3*B/(4*a_13**2*a_14**2) * (3/(2*omega+a_14**3)*np.sin((2*omega+a_14**3)*t + 2*delta - a_14**2*b_14) + 1/(2*omega-a_14**3)*np.sin((2*omega-a_14**3)*t + 2*delta + a_14**2*b_14)) )
                               + 3*a_13**2 * ( d15/(omega+a_14**3)**2*np.cos((omega+a_14**3)*t + delta - a_14**2*b_14) + d16/(omega-a_14**3)**2*np.cos((omega-a_14**3)*t + delta + a_14**2*b_14) - 3*A**3*B/(4*a_13**2*a_14**2) * (3/(2*omega+a_14**3)**2*np.cos((2*omega+a_14**3)*t + 2*delta - a_14**2*b_14) + 1/(2*omega-a_14**3)**2*np.cos((2*omega-a_14**3)*t + 2*delta + a_14**2*b_14)) )
                              )
                      )
    b1_13 = b1_13_func(t) - b1_13_func(0)

    a1_23_func = lambda t: m4 * ( -c/omega * np.cos(omega*t + delta) + sum_1(t)/omega
                       - e3 * (d17/(omega+a_13**3)*np.sin((omega+a_13**3)*t + delta - a_13**2*b_13) + d18/(omega-a_13**3)*np.sin((omega-a_13**3)*t + delta + a_13**2*b_13) + 3*A**3*B/(4*a_13**2*a_14**2) * (1/(2*omega + a_13**3)*np.sin((2*omega+a_13**3)*t + 2*delta - a_13**2*b_13) + 3/(2*omega - a_13**3)*np.sin((2*omega-a_13**3)*t + 2*delta + a_13**2*b_13)))
                       - e4 * (d15/(omega+a_14**3)*np.sin((omega+a_14**3)*t + delta - a_14**2*b_14) + d16/(omega-a_14**3)*np.sin((omega-a_14**3)*t + delta + a_14**2*b_14) - 3*A**3*B/(4*a_13**2*a_14**2) * (3/(2*omega + a_14**3)*np.sin((2*omega+a_14**3)*t + 2*delta - a_14**2*b_14) + 1/(2*omega - a_14**3)*np.sin((2*omega-a_14**3)*t + 2*delta + a_14**2*b_14)))
                      )
    a1_23 = a1_23_func(t) - a1_23_func(0)

    b1_23_func = lambda t: m4 * ( c/(omega*a_23)*np.sin(omega*t+delta)
                       + 1/2*e3*a_13**2*a_14**2*a_24 * (3/(omega+a_13**3)*np.cos((omega+a_13**3)*t + delta - a_13**2*b_13) + 1/(omega-a_13**3)*np.cos((omega-a_13**3)*t + delta + a_13**2*b_13))
                       - 1/2*e4*a_13**2*a_14 * ((a_14*a_24 - 1)/(omega+a_14**3)*np.cos((omega+a_14**3)*t + delta - a_14**2*b_14) + (3*a_14*a_24 - 1)/(omega-a_14**3)*np.cos((omega-a_14**3)*t + delta + a_14**2*b_14))
                      )
    b1_23 = b1_23_func(t) - b1_23_func(0)
    #print(a1_13_func(t), b1_13_func(t), a1_23_func(t), b1_23_func(t))
    #print(a1_13_func(0), b1_13_func(0), a1_23_func(0), b1_23_func(0))
    #print(a1_13, b1_13, a1_23, b1_23)

    return 1/22903 * ( (3*a_13**2*t - 2*a_13*b_13)*a1_13 + b1_23 - a_13**2*b1_13
                       -2*np.sqrt(1 - a_13**2*a_23**2) * ((3*a_13**2*t - 2*a_13*b_13)*a1_13 - a_13**2*b1_13) * np.sin(a_13**3*t - a_13**2*b_13)
                       -2*(a_13*a_23**2*a1_13 + a_13**2*a_23*a1_23) / np.sqrt(1 - a_13**2*a_23**2) * np.cos(a_13**3*t - a_13**2*b_13)
                      )

popt, pcov = curve_fit(phi1_13, temps, err, p0 = (1.180, 1.567, 0.005, 6.470, 0.998, 0, 0, 0, 0), bounds=((1, 1.1, 0, 0, 0, -1, -0.2, -10, -10), (2, 3, 0.1, 10, 2*np.pi, 1, 0.2, 10, 10)))
print(popt)

# Affichage de l'erreur en fonction du temps
def tta(temps):
    return temps * 84.02 / (2*np.pi) + 1800
temps2 = np.linspace(np.min(temps), np.max(temps), 1000)
plt.style.use ("https://raw.githubusercontent.com/HunsterMonter/ggplot-dark/main/ggplot_dark.mplstyle")
plt.figure (layout="constrained")
plt.plot(tta(temps), err / np.pi*180 * 3600, ".", color="tab:blue")
plt.xlabel("Année")
plt.ylabel('Erreur sur la longitude (")')
plt.savefig("Erreurs.png", dpi=600)
plt.plot(tta(temps2), phi1_13(temps2, *popt) / np.pi*180 * 3600, color="tab:red")
plt.savefig("Régression.png", dpi=600)

print(phi1_13(10, 1.180, 1.567, 0.007, 1.5, 0.998, 0, 0, 0, 0))
# Décourverte le 23 septembre 1846
t = 3.49413
m4, a4, e4, t04, phi04, h1, h2, h3, h4 = popt
#var = pcov[1,1]**2 + 9/4 * t**2/a**5 * pcov[0,0]**2 - 3*t/a**2.5 * pcov[0,1]
print(f"Le jour de sa découverte, la longitude de Neptune serait de {180 / np.pi * (phi04 + (t-t04) / a4**1.5 + 2*e4*np.sin((t-t04) / a4**1.5)):.1f} ± {180 / np.pi * 0**0.5:.1f} degrés")
plt.show()
