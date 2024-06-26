from calendar import isleap
from datetime import date, datetime
from math import modf
from scipy.optimize import curve_fit
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


# Temps pour t<1821
temps = np.array([-0.8915, -0.7579, -0.7267, -0.2837, -0.2817, -0.2525, -0.2276, -0.1630, -0.1201, -0.0957, 0.0076, 0.0100, 0.0123, 0.0180, 0.0205, 0.0227, 0.0288, 0.0313, 0.0335, 0.0393, 0.0421, 0.0442, 0.0508, 0.0530, 0.0554, 0.0640, 0.0752, 0.0769, 0.0862, 0.0886, 0.0975, 0.1063, 0.1088, 0.1174, 0.1201, 0.1287, 0.1315, 0.1438, 0.1513, 0.1541, 0.1626, 0.1654, 0.1740, 0.1767, 0.1852, 0.1877, 0.1989, 0.2068, 0.2098, 0.2206, 0.2290, 0.2314, 0.2433, 0.2523, 0.2626, 0.2703, 0.2725, 0.2800, 0.2835, 0.2922, 0.3015, 0.3106, 0.3179, 0.3198, 0.3268, 0.3287, 0.3354, 0.3373, 0.3444, 0.3458, 0.3519, 0.3539, 0.3600, 0.3618, 0.3697, 0.3781, 0.3878, 0.3926, 0.3996])
cosphi = 0.0001 * np.array([-1888, 9856, 9988, -9089, -9130, -9809, -9999, -8470, -5829, -4058, 3087, 3248, 3387, 3790, 3954, 4077, 4477, 4638, 4794, 5136, 5293, 5443, 5798, 5920, 6040, 6512, 7069, 7155, 7579, 7685, 8054, 8481, 8479, 8771, 8857, 9111, 9182, 8456, 9627, 9674, 9804, 9838, 9926, 9945, 9990, 9996, 9989, 9948, 9924, 9804, 9670, 9625, 9393, 9104, 8768, 8475, 8373, 8053, 7943, 7460, 6936, 6369, 5904, 5763, 5284, 5121, 4619, 4456, 3946, 3769, 3237, 3075, 2525, 2335, 1598, 843, 91, -640, -1390])
e = 0.0466108
temps *= 100 / (1 + 2*e*cosphi) # Temps en années depuis 1781
temps -= 19 # Temps en années depuis 1800

# Temps pour t>1821
annee_next = np.array ([1833, 1833, 1833, 1833, 1833, 1833, 1834, 1834, 1834, 1834, 1834, 1835, 1835, 1835, 1835, 1835, 1835, 1836, 1836, 1836, 1836, 1836, 1836])
jour_next = np.array ([215, 235, 265, 279, 295, 323, 229, 258, 283, 312, 336, 215, 232, 255, 293, 326, 337, 203, 236, 256, 285, 307, 327])
epoch = datetime (1800, 1, 1, 0)
leap = np.array ([isleap (a) for a in annee_next])
annee_dec_next = np.where (leap, annee_next + jour_next/366, annee_next + jour_next/365)

# Combine les deux arrays de temps
temps = np.concatenate((temps, np.array([(dec_to_date (an) - epoch).days for an in annee_dec_next])/365.25))

# Temps en omega_3 depuis le jour de sa découverte
temps = temps * 2*np.pi / 84.02 - 3.49413

# Erreurs sur les tables
err = np.pi/(200*10000) * np.array ([132.7, 202.7, 135.7, -227.7, -206.4, -186.6, -203.5, -151.5, -99.1, -46.9, -17.2, -10.3, -4.7, -10.3, -13.0, -1.0, 2.7, -3.8, 0.4, -9.0, -7.7, -0.5, 2.3, 2.1, 5.2, -10.8, 11.7, 2.9, 32.2, 24.0, -4.6, 4.0, 12.0, 12.7, 4.9, 4.2, 10.8, 8.3, 8.2, 3.3, 16.4, -1.9, 12.8, -5.9, -5.3, -20.3, -12.8, 6.3, -12.5, -19.2, -4.9, -15.1, -2.3, -14.5, -2.7, 8.5, -10.6, 11.6, -2.1, -13.3, -4.5, 3.8, -9.4, -5.9, 4.7, -2.4, 17.8, 1.6, -11.9, 7.7, 8.0, 8.7, 9.7, 2.8, -2.1, 9.4, -0.6, 0.3, -4.2])
err_next = -np.pi/(180*3600) * (np.array ([31.20, 31.53, 32.03, 32.27, 32.53, 33.00, 37.47, 37.97, 38.41, 38.92, 39.34, 43.88, 44.21, 44.66, 45.40, 46.05, 46.26, 51.06, 51.73, 52.14, 52.74, 53.19, 53.60]) + 0.00337 * np.array ([284, 286, 290, 291, 293, 296, 337, 335, 333, 331, 329, 248, 245, 241, 234, 228, 226, 210, 207, 205, 203, 201, 199]))
err = np.concatenate ((err, err_next))

# Temps et erreur selon Le Verrier
# tempsV = np.array([1690.98, 1712.25, 1715.23, 1747.7, 1754.7, 1761.7, 1768.7, 1775.7, 1782.7, 1789.7, 1796.7, 1803.7, 1810.7, 1817.7, 1824.7, 1831.7, 1838.7, 1845.7])
# tempsV = (tempsV - 1800) * 2*np.pi / 84.02 - 3.49413
# errV = np.pi / (180 * 3600) * np.array([-63.1, -59.9, -64.6, 34.8, 32.8, 24.7, 10.0, -3.7, -17.4, -28.6, -29.8, -33.6, -35.3, -32.3, -24.5, 3.4, 50.0, 110.5])

# Fonction à régresser sur l'erreur des tables
def phi1_13(t, m4, a4, phi_04, h1, h2, h3, h4):
    # On pose l'excentricité de Neptune nulle afin d'éviter le surajustement
    e4 = 0
    t_04 = 0

    # Éléments orbitaux de Neptune
    a_14 = 1 / a4**0.5
    a_24 = (a4 * (1-e4**2))**0.5
    b_14 = t_04 / a4**0.5 + np.pi/2 * a4
    b_24 = phi_04
    
    # Constantes
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

    c0 = A*(1 + 3*B**2/16 + 105*B**4/2**10 + 1155*B**6/2**14 + 255255*B**8/2**22)
    c1 = A*(B/2 + 15*B**3/64 + 315*B**5/2**11 + 15015*B**7/2**17 + 765765*B**9/2**23)
    c2 = A*(3*B**2/16 + 35*B**4/2**8 + 3465*B**6/2**15 + 45045*B**8/2**19)
    c3 = A*(5*B**3/64 + 315*B**5/2**12 + 9009*B**7/2**17 + 255255*B**9/2**22)
    c4 = A*(35*B**4/2**10 + 693*B**6/2**14 + 45045*B**8/2**20)
    c5 = A*(63*B**5/2**12 + 3003*B**7/2**17 + 109395*B**9/2**22)
    c6 = A*(231*B**6/2**15 + 6435*B**8/2**19)
    c7 = A*(429*B**7/2**17 + 109395*B**9/2**24)
    c8 = A*(6435*B**8/2**22)
    c9 = A*(12155*B**9/2**24)

    cn = (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)

    pdvC0 = A*(2*3*B/16 + 4*105*B**3/2**10 + 6*1155*B**5/2**14 + 8*255255*B**7/2**22)
    pdvC1 = A*(1/2 + 3*15*B**2/64 + 5*315*B**4/2**11 + 7*15015*B**6/2**17 + 9*765765*B**8/2**23)
    pdvC2 = A*(2*3*B/16 + 4*35*B**3/2**8 + 6*3465*B**5/2**15 + 8*45045*B**7/2**19)
    pdvC3 = A*(3*5*B**2/64 + 5*315*B**4/2**12 + 7*9009*B**6/2**17 + 9*255255*B**8/2**22)
    pdvC4 = A*(4*35*B**3/2**10 + 6*693*B**5/2**14 + 8*45045*B**7/2**20)
    pdvC5 = A*(5*63*B**4/2**12 + 7*3003*B**6/2**17 + 9*109395*B**8/2**22)
    pdvC6 = A*(6*231*B**5/2**15 + 8*6435*B**7/2**19)
    pdvC7 = A*(7*429*B**6/2**17 + 9*109395*B**8/2**24)
    pdvC8 = A*(8*6435*B**7/2**22)
    pdvC9 = A*(9*12155*B**8/2**24)

    pdvCn = (pdvC0, pdvC1, pdvC2, pdvC3, pdvC4, pdvC5, pdvC6, pdvC7, pdvC8, pdvC9)

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

    # Perburbations sur les nouvelles variables canoniques
    a1_13_func = lambda t: m4 * (
        a_13**2/omega * ( c*np.cos(omega*t + delta) - sum_1(t) )
        - e3 * (d1/a_13**3 * np.sin(a_13**3*t - a_13**2*b_13) + d2/(omega+a_13**3)*np.sin((omega+a_13**3)*t + delta - a_13**2*b_13) - 9*A**3*B/(8*a_14**2) * (1/(2*omega+a_13**3)*np.sin((2*omega+a_13**3)*t + 2*delta - a_13**2*b_13) + 1/(2*omega-a_13**3)*np.sin((2*omega-a_13**3)*t + 2*delta + a_13**2*b_13)))
    )
    """
        - e4 * (d3/(omega+a_14**3)*np.sin((omega+a_14**3)*t + delta - a_14**2*b_14) + d4/(omega-a_14**3)*np.sin((omega-a_14**3)*t + delta + a_14**2*b_14) + 3*A**3*B/(4*a_14**2) * (3/(2*omega+a_14**3)*np.sin((2*omega+a_14**3)*t + 2*delta - a_14**2*b_14) + 1/(2*omega-a_14**3)*np.sin((2*omega-a_14**3)*t + 2*delta + a_14**2*b_14)))
    )
    """
    a1_13 = a1_13_func(t) - a1_13_func(0) + h1

    b1_13_func = lambda t: m4 * (
        2*c/(a_13*omega) * np.sin(omega*t + delta) - (cn[0]*pdvA/A + pdvCn[0]*pdvB)*t - sum_3(t)/omega - 3*a_13**2/omega**2 * (c*np.sin(omega*t + delta) - sum_2(t)) + (3*a_13**2*t - 2*a_13*b_13)/omega * (c*np.cos(omega*t + delta) - sum_1(t))
        + e3 * ( - d5/a_13**3*np.cos(a_13**3*t - a_13**2*b_13) - d6/(omega+a_13**3)*np.cos((omega+a_13**3)*t + delta - a_13**2*b_13) - d7/(omega-a_13**3)*np.cos((omega-a_13**3)*t + delta + a_13**2*b_13) - d8/(2*omega+a_13**3)*np.cos((2*omega+a_13**3)*t + 2*delta - a_13**2*b_13) - 3*d8/(2*omega-a_13**3)*np.cos((2*omega-a_13**3)*t + 2*delta + a_13**2*b_13)
                 + (3*a_13**2*t - 2*a_13*b_13) * ( d9/a_13**3*np.sin(a_13**3*t - a_13**2*b_13) + d10/(omega+a_13**3)*np.sin((omega+a_13**3)*t + delta - a_13**2*b_13) + 9*A**3*B/(8*a_13**2*a_14**2) * (1/(2*omega+a_13**3)*np.sin((2*omega+a_13**3)*t + 2*delta - a_13**2*b_13) + 1/(2*omega-a_13**3)*np.sin((2*omega-a_13**3)*t + 2*delta + a_13**2*b_13)) )
                 + 3*a_13**2 * ( d9/a_13**6*np.cos(a_13**3*t - a_13**2*b_13) + d10/(omega+a_13**3)**2*np.cos((omega+a_13**3)*t + delta - a_13**2*b_13) + 9*A**3*B/(8*a_13**2*a_14**2) * (1/(2*omega+a_13**3)**2*np.cos((2*omega+a_13**3)*t + 2*delta - a_13**2*b_13) + 1/(2*omega-a_13**3)**2*np.cos((2*omega-a_13**3)*t + 2*delta + a_13**2*b_13)) )
               )
    )
    """
        + e4 * ( - d11/a_14**3*np.cos(a_14**3*t - a_14**2*b_14) - d12/(omega+a_14**3)*np.cos((omega+a_14**3)*t + delta - a_14**2*b_14) - d13/(omega-a_14**3)*np.cos((omega-a_14**3)*t + delta + a_14**2*b_14) - 3*d14/(2*omega+a_14**3)*np.cos((2*omega+a_14**3)*t + 2*delta - a_14**2*b_14) - d14/(2*omega-a_14**3)*np.cos((2*omega-a_14**3)*t + 2*delta + a_14**2*b_14)
                 + (3*a_13**2*t - 2*a_13*b_13) * ( d15/(omega+a_14**3)*np.sin((omega+a_14**3)*t + delta - a_14**2*b_14) + d16/(omega-a_14**3)*np.sin((omega-a_14**3)*t + delta + a_14**2*b_14) - 3*A**3*B/(4*a_13**2*a_14**2) * (3/(2*omega+a_14**3)*np.sin((2*omega+a_14**3)*t + 2*delta - a_14**2*b_14) + 1/(2*omega-a_14**3)*np.sin((2*omega-a_14**3)*t + 2*delta + a_14**2*b_14)) )
                 + 3*a_13**2 * ( d15/(omega+a_14**3)**2*np.cos((omega+a_14**3)*t + delta - a_14**2*b_14) + d16/(omega-a_14**3)**2*np.cos((omega-a_14**3)*t + delta + a_14**2*b_14) - 3*A**3*B/(4*a_13**2*a_14**2) * (3/(2*omega+a_14**3)**2*np.cos((2*omega+a_14**3)*t + 2*delta - a_14**2*b_14) + 1/(2*omega-a_14**3)**2*np.cos((2*omega-a_14**3)*t + 2*delta + a_14**2*b_14)) )
               )
        )
    """
    b1_13 = b1_13_func(t) - b1_13_func(0) + h2

    a1_23_func = lambda t: m4 * (
        - c/omega * np.cos(omega*t + delta) + sum_1(t)/omega
        - e3 * (d17/(omega+a_13**3)*np.sin((omega+a_13**3)*t + delta - a_13**2*b_13) + d18/(omega-a_13**3)*np.sin((omega-a_13**3)*t + delta + a_13**2*b_13) + 3*A**3*B/(4*a_13**2*a_14**2) * (1/(2*omega + a_13**3)*np.sin((2*omega+a_13**3)*t + 2*delta - a_13**2*b_13) + 3/(2*omega - a_13**3)*np.sin((2*omega-a_13**3)*t + 2*delta + a_13**2*b_13)))
    )
    """
        - e4 * (d15/(omega+a_14**3)*np.sin((omega+a_14**3)*t + delta - a_14**2*b_14) + d16/(omega-a_14**3)*np.sin((omega-a_14**3)*t + delta + a_14**2*b_14) - 3*A**3*B/(4*a_13**2*a_14**2) * (3/(2*omega + a_14**3)*np.sin((2*omega+a_14**3)*t + 2*delta - a_14**2*b_14) + 1/(2*omega - a_14**3)*np.sin((2*omega-a_14**3)*t + 2*delta + a_14**2*b_14)))
    )
    """
    a1_23 = a1_23_func(t) - a1_23_func(0) + h3
    
    b1_23_func = lambda t: m4 * (
        c/(omega*a_23)*np.sin(omega*t+delta)
        + 1/2*e3*a_13**2*a_14**2*a_24 * (3/(omega+a_13**3)*np.cos((omega+a_13**3)*t + delta - a_13**2*b_13) + 1/(omega-a_13**3)*np.cos((omega-a_13**3)*t + delta + a_13**2*b_13))
    )
    """
        - 1/2*e4*a_13**2*a_14 * ((a_14*a_24 - 1)/(omega+a_14**3)*np.cos((omega+a_14**3)*t + delta - a_14**2*b_14) + (3*a_14*a_24 - 1)/(omega-a_14**3)*np.cos((omega-a_14**3)*t + delta + a_14**2*b_14))
    )
    """
    b1_23 = b1_23_func(t) - b1_23_func(0) + h4

    # Perburbation sur la longitude d'Uranus
    return 1/22903 * ( (3*a_13**2*t - 2*a_13*b_13)*a1_13 + b1_23 - a_13**2*b1_13
                       -2*np.sqrt(1 - a_13**2*a_23**2) * ((3*a_13**2*t - 2*a_13*b_13)*a1_13 - a_13**2*b1_13) * np.sin(a_13**3*t - a_13**2*b_13)
                       -2*(a_13*a_23**2*a1_13 + a_13**2*a_23*a1_23) / np.sqrt(1 - a_13**2*a_23**2) * np.cos(a_13**3*t - a_13**2*b_13)
                     )


# Éléments orbitaux d'Uranus
a3 = 1
e3 = 0.047
t_03 = 6.155 - 3.49413
phi_03 = 2.939

a_13 = 1 / a3**0.5
a_23 = (a3 * (1-e3**2))**0.5
b_13 = t_03 / a3**0.5 + np.pi/2 * a3
b_23 = phi_03

# Régression
popt, pcov = curve_fit(phi1_13, temps, err, p0 = (1.180, 1.567, 0, 0, 0, 0, 0), bounds=((1, 1.1, -np.inf, -10000, -10000, -10000, -10000), (2, 2, np.inf, 10000, 10000, 10000, 10000)))
#popt, pcov = curve_fit(phi1_13, tempsV, -errV, p0 = (1.180, 1.567, 0.007, 6.470, 0.998, 0, 0, 0, 0), bounds=((1.180, 1.567, 0.007, 6.470, 0.998, -10000, -10000, -10000, -10000), (1.1801, 1.5671, 0.0071, 6.4701, 0.9981, 10000, 10000, 10000, 10000)))
print(popt)

# Affichage de l'erreur en fonction du temps
def tta(temps):
    return (temps + 3.49413) * 84.02 / (2*np.pi) + 1800

# Graphiques
temps2 = np.linspace(np.min(temps), np.max(temps), 10000)
plt.style.use ("ggplot")
plt.figure (layout="constrained")
plt.plot(tta(temps), err / np.pi*180 * 3600, ".", color="tab:blue")
plt.xlabel("Année")
plt.ylabel('Erreur sur la longitude (")')
plt.savefig("Erreurs.png", dpi=600)
plt.plot(tta(temps2), phi1_13(temps2, *popt) / np.pi*180 * 3600, color="tab:red")
plt.savefig("Régression.png", dpi=600)

# Décourverte le 23 septembre 1846
m4, a4, phi04, h1, h2, h3, h4 = popt
print(f"Le jour de sa découverte, la longitude de Neptune serait de {180 / np.pi * phi04 % 360:.1f} ± {180 / np.pi * pcov[2,2]**0.5:.1f} degrés.")
plt.show()
