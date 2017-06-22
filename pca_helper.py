import numpy as np
import sklearn.decomposition as decomposition


def get_pca(x, n_comoponents=4, whiten=True):
    pca = decomposition.PCA(n_components=n_comoponents)
    pca.fit(x)

    return pca


def get_pca_coeff(x, n_components=4):
    x2 = np.nan_to_num(x)
    x3 = x2.reshape((x2.shape[0], -1))

    pca = get_pca(x3, n_components)
    # print(pca.explained_variance_ratio_)

    # PCA(copy=True, n_components=2, whiten=False)
    # accumulated_var = np.add.accumulate(pca.explained_variance_ratio_)
    pc = pca.transform(x3)

    return pca, pc


def apply_pca(pca, x):
    x2 = np.nan_to_num(x)
    x3 = x2.reshape((x2.shape[0], -1))

    return pca.transform(x3)


def show_components(pca_list, fig=None, imshow_kwargs=None):
    from mpl_toolkits.axes_grid1 import ImageGrid

    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 12))

    if imshow_kwargs is None:
        imshow_kwargs = dict(interpolation="none", cmap="jet")

    for i, p in enumerate(pca_list):
        components = p.components_
        shape0 = int(np.sqrt(len(components[0])))

        grid = ImageGrid(fig, (len(pca_list), 1, i + 1),
                         (1, len(components)), axes_pad=0.1)

        for ax, c in zip(grid, components):
            ax.imshow(np.reshape(c, (shape0, shape0)), **imshow_kwargs)


# pca_list = [pcaC, pca0, pca10, pca11, pca12, pca20, pca21, pca22]
# show_components(pca_list)
