import pandas as pd
import numpy as np

from visualisation import *


file = "results/Q_val_policy_Q0_val_space_3.csv"
df = pd.read_csv(file)

VIZ = Visualisation(data_path=file)
VIZ.tabQ_policy_2_3D()

