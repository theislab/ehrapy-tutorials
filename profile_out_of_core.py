import scalene

scalene.scalene_profiler.stop()

import dask.array as da
from sklearn.datasets import make_blobs as make_blobs
import ehrapy as ep
import anndata as ad
import pandas as pd

n_individuals = 50000
n_features = 1000
n_groups = 4
chunks = 1000

data_features, data_labels = make_blobs(
    n_samples=n_individuals, n_features=n_features, centers=n_groups, random_state=42
)

data_features = da.from_array(data_features, chunks=chunks)

var = pd.DataFrame({"feature_type": ["numeric"] * n_features})

adata = ad.AnnData(X=data_features, obs={"label": data_labels}, var=var)

scalene.scalene_profiler.start()

ep.pp.scale_norm(adata)

ep.pp.pca(adata)

adata.obsm["X_pca"] = adata.obsm["X_pca"].compute()

ep.pp.neighbors(adata)

ep.tl.leiden(adata)

ep.pl.pca(adata, color="leiden", save="profiling_out_of_core_pca.png")

scalene.scalene_profiler.stop()
