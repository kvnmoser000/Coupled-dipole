import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Constants
hbar=6.582119569e-16
folder = "data_tw"

plt.figure(figsize=(9,6))

if folder == "data":
    pattern=re.compile(r"dipole_data_(\d{2})d\.npz")
elif folder == "data_tw_close":
    pattern=re.compile(r"dipole_data_(\d{4}).npz")
elif folder == "data_tw":
    pattern=re.compile(r"dipole_data_(\d{4}).npz")

for fname in sorted(os.listdir(folder)):
    match=pattern.match(fname)
    if match:
        raw_num=match.group(1)  # e.g., "23"
        if folder == "data":
            h_str=f"{raw_num[0]}.{raw_num[1]}"  # → "2.3"
        elif folder =="data_tw_close":
            h_str=f"{int(raw_num[:-2])}.{raw_num[-2:]}"
        elif folder =="data_tw":
            h_str=f"{int(raw_num[:-2])}.{raw_num[-2:]}"
        filepath=os.path.join(folder,fname)

        data=np.load(filepath)
        eigenvalues=data["eigenvalues"]
        energies=np.real(np.sqrt(eigenvalues)*hbar)

        df=pd.DataFrame({"Energy (eV)":energies})
        sns.kdeplot(data=df,x="Energy (eV)",fill=False,label=fr"$\theta$ = {h_str}")

plt.xlabel("Energy (eV)")
plt.ylabel("Density of States")
plt.legend(title="Twist Angle")
plt.title("Bilayer Density of States")
plt.tight_layout()
plt.show()
