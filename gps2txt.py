import pymap3d as pm
import pandas as pd
import numpy as np

# TODO: .gpx to .csv here

df = pd.read_csv("track.csv")

lat = np.array(df.iloc[:394, 0])
lon = np.array(df.iloc[:394, 1])

x,y,z = pm.geodetic2ecef(lat, lon, alt=65)

x = x - x[0]
y = y - y[0]
z = z - z[0]

x = x / 1710 * 2140
y = y / 1710 * 2140
z = z / 1710 * 2140

mats = []
for xi, yi, zi in zip(x, y, z):
    yi = 0
    mats.append([0, 0, 0, -xi, 0, 0, 0, -yi, 0, 0, 0, zi])

with open("track.txt", "w") as f:
    for mat in mats:
        f.write(str(mat).strip("[]").replace(",", "") + "\n")

