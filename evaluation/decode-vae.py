import pandas as pd

gene_id_vae = pd.read_csv("../data/yeast_emb_embeddings_yeastnet_genex.csv").gene_id
gene_id_cvae = pd.read_parquet("../data/smf30_filtered_emb.parquet").gene_id

vae = pd.read_parquet("../data/vae_embeddings.parquet")
cvae = pd.read_parquet("../data/cvae_embeddings.parquet")

vae.set_index(gene_id_vae).to_csv("../data/vae.csv")
cvae.set_index(gene_id_cvae).to_csv("../data/cvae.csv")
