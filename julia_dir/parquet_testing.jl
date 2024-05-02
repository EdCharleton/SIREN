
using Tables, DataFrames
using PyCall
using DuckDB

matplotlib = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
rcParams = matplotlib[:rcParams]
path = "sn_template_dir/type_Ia/salt3_-1.0_-0.1_0.0.parquet"

con = DBInterface.connect(DuckDB.DB)
results = DBInterface.execute(con,  """SELECT * FROM read_parquet('$path')""")
df = DataFrame((results))
array = collect(df.spectra)
