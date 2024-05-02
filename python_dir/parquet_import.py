import pyarrow.parquet as pq

def pyquet(path):
    return pq.read_table(path).to_pandas()


parquet_path = "sn_template_dir/type_Ia/salt3_-1.0_-0.1_0.0.parquet"

file = pyquet(parquet_path)

print(file.head())



