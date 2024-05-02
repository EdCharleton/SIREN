import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pyarrow.parquet as pq
import pyarrow as pa

parquet_file_path = "sn_template_dir/type_Ia/salt3_-1.0_-0.1_0.0.parquet"

# Read the parquet file
df = pq.read_table(parquet_file_path).to_pandas()

print(df)

y_vals = df['spectra'].values
x_vals = np.arange(len(y_vals))







