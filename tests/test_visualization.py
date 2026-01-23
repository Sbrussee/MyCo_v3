import numpy as np

from myco.visualization import _project_embeddings


def test_project_embeddings_tsne_shape():
    embeddings = np.random.RandomState(0).rand(64, 8)
    coords = _project_embeddings(embeddings, method="tsne", random_state=0)
    assert coords.shape == (64, 2)
