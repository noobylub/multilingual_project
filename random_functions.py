import pandas as pd

# For loading data, we can load multiple languages 
def return_data(language:str):
    return pd.read_csv(f"hf://datasets/mrlbenchmarks/global-piqa-parallel/data/parallel_{language}.tsv", sep="\t")

