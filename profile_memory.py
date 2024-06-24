import scalene

scalene.scalene_profiler.stop()

import pandas as pd
from sklearn.datasets import make_blobs as make_blobs
import ehrapy as ep
import anndata as ad
import scanpy as sc

n_individuals = 50000
n_features = 1000
n_groups = 4
chunks = 1000

data_features, data_labels = make_blobs(n_samples=n_individuals, n_features=n_features, centers=n_groups, random_state=42)

var = pd.DataFrame({"feature_type": ["numeric"] * n_features})

adata = ad.AnnData(X=data_features, obs={"label": data_labels}, var=var)

scalene.scalene_profiler.start()

ep.pp.scale_norm(adata)

ep.pp.pca(adata)

sc.pp.neighbors(adata)

ep.tl.leiden(adata)

ep.pl.pca(adata, color="leiden", save="profiling_memory_pca.png")

scalene.scalene_profiler.stop()
