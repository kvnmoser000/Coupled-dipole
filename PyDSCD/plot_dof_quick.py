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



filepath=os.path.join("dipole_data.npz")

data=np.load(filepath)
eigenvalues=data["eigenvalues"]
energies=np.real(np.sqrt(eigenvalues)*hbar)

df=pd.DataFrame({"Energy (eV)":energies})
sns.kdeplot(data=df,x="Energy (eV)",fill=False)

plt.xlabel("Energy (eV)")
plt.ylabel("Density of States")
plt.title("Density of States")
plt.tight_layout()
plt.show()