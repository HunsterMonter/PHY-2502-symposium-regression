from calendar import isleap
from datetime import date, datetime, timedelta
from functools import partial
from math import modf
from scipy.odr import ODR, Model, RealData
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import numpy as np


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


def f (beta, x):
    y = beta[0]*x + beta[1]

    for i in range (3):
        y += beta[3*i+2] * np.sin (2*np.pi*x/beta[3*i+3] + beta[3*i+4])

    return y


# Données brutes
annee_prev = np.array ([1690, 1712, 1715, 1749.25, 1750.25, 1753, 1756, 1764, 1769, 1771]) + 0.5
annee = np.array ([1781, 1781, 1782, 1782, 1782, 1783, 1783, 1783, 1784, 1784, 1785, 1785, 1785, 1786, 1786, 1787, 1788, 1788, 1789, 1789, 1790, 1790, 1791, 1791, 1792, 1792, 1793, 1794, 1794, 1795, 1795, 1796, 1796, 1797, 1797, 1798, 1799, 1799, 1800, 1801, 1802, 1802, 1803, 1804, 1805, 1806, 1806, 1807, 1807, 1808, 1809, 1810, 1811, 1811, 1812, 1812, 1813, 1813, 1814, 1814, 1815, 1815, 1816, 1816, 1817, 1818, 1819, 1820, 1821]) + 0.5
annee_next = np.array ([1833, 1833, 1833, 1833, 1833, 1833, 1834, 1834, 1834, 1834, 1834, 1835, 1835, 1835, 1835, 1835, 1835, 1836, 1836, 1836, 1836, 1836, 1836])
jour_next = np.array ([215, 235, 265, 279, 295, 323, 229, 258, 283, 312, 336, 215, 232, 255, 293, 326, 337, 203, 236, 256, 285, 307, 327])
sigma_annee = 0.25

t = np.array ([0.0076, 0.0100, 0.0123, 0.0180, 0.0205, 0.0227, 0.0288, 0.0313, 0.0335, 0.0393, 0.0421, 0.0442, 0.0508, 0.0530, 0.0554, 0.0640, 0.0752, 0.0769, 0.0862, 0.0886, 0.0975, 0.1063, 0.1088, 0.1174, 0.1201, 0.1287, 0.1315, 0.1438, 0.1513, 0.1541, 0.1626, 0.1654, 0.1740, 0.1767, 0.1852, 0.1877, 0.1989, 0.2068, 0.2098, 0.2206, 0.2290, 0.2314, 0.2433, 0.2523, 0.2626, 0.2703, 0.2725, 0.2800, 0.2835, 0.2922, 0.3015, 0.3106, 0.3179, 0.3198, 0.3268, 0.3287, 0.3354, 0.3373, 0.3444, 0.3458, 0.3519, 0.3539, 0.3600, 0.3618, 0.3697, 0.3781, 0.3878, 0.3926, 0.3996])
sigma_t = 0.00005

err_prev = np.pi/(200*10000) * np.array ([132.7, 202.7, 135.7, -227.7, -206.4, -186.6, -203.5, -151.5, -99.1, -46.9])
err = np.pi/(200*10000) * np.array ([-17.2, -10.3, -4.7, -10.3, -13.0, -1.0, 2.7, -3.8, 0.4, -9.0, -7.7, -0.5, 2.3, 2.1, 5.2, -10.8, 11.7, 2.9, 32.2, 24.0, -4.6, 4.0, 12.0, 12.7, 4.9, 4.2, 10.8, 8.3, 8.2, 3.3, 16.4, -1.9, 12.8, -5.9, -5.3, -20.3, -12.8, 6.3, -12.5, -19.2, -4.9, -15.1, -2.3, -14.5, -2.7, 8.5, -10.6, 11.6, -2.1, -13.3, -4.5, 3.8, -9.4, -5.9, 4.7, -2.4, 17.8, 1.6, -11.9, 7.7, 8.0, 8.7, 9.7, 2.8, -2.1, 9.4, -0.6, 0.3, -4.2])
err_next = -np.pi/(180*3600) * (np.array ([31.20, 31.53, 32.03, 32.27, 32.53, 33.00, 37.47, 37.97, 38.41, 38.92, 39.34, 43.88, 44.21, 44.66, 45.40, 46.05, 46.26, 51.06, 51.73, 52.14, 52.74, 53.19, 53.60]) + 0.00337 * np.array ([284, 286, 290, 291, 293, 296, 337, 335, 333, 331, 329, 248, 245, 241, 234, 228, 226, 210, 207, 205, 203, 201, 199]))

# Erreur en rad
err = np.concatenate ((err_prev[3:], err, err_next))

# Régression pour obtenir le temps décimal
data = RealData (annee, t, sigma_annee, sigma_t)
model = Model (f)

odr = ODR (data, model, [0.01, -17.8, 0, 400, 0, 0, 200, 0, 0, 100, 0])
odr.set_job (fit_type=2)
output = odr.run ()

# Fonctions utilisées pour trouver le temps décimal
f_opt = partial (f, output.beta)
f_t = lambda t, x: f_opt (x) - t
f_v = [partial (f_t, i) for i in t]
x0 = [(i-output.beta[1]) / output.beta[0] for i in t]

# Calcule l'année décimale des données
annee_dec = np.array ([root_scalar (f_i, x0=xi).root for xi, f_i in zip(x0, f_v)])
leap = np.array ([isleap (a) for a in annee_next])
annee_dec_next = np.where (leap, annee_next + jour_next/366, annee_next + jour_next/365)
annee_dec = np.concatenate ((annee_prev[3:], annee_dec, annee_dec_next))

# Affiche le nombre de points qui sont mal repésentés par la régression
print (np.sum ([abs (root_scalar (f_i, x0=xi).root - a) > 0.5 for xi, f_i, a in zip(x0, f_v, annee)]))
# Affiche les paramètres de la régression
print (output.beta)

# Date de référence
epoch = datetime (1800, 1, 1, 0)
# Calcule le nombre de jours depuis le 1er janvier 1800
jours_ecoules = np.array([(dec_to_date (an) - epoch).days for an in annee_dec], int)
# Temps en 2*pi/période
temps = jours_ecoules/365.25 * 2*np.pi/84.02

xn = np.linspace (1781, 1822, 1000)
yn = f_opt (xn)

# Affichage de la régression
plt.style.use ("https://raw.githubusercontent.com/HunsterMonter/ggplot-dark/main/ggplot_dark.mplstyle")
plt.figure (layout="constrained")
plt.scatter (annee, t, c="#1374ad")
plt.plot (xn, yn)
plt.show ()

# Affichage de l'erreur en fonction du temps
plt.figure (layout="constrained")
plt.scatter (temps, err)
plt.show ()
