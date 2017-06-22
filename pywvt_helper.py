from operator import itemgetter
import numpy as np


def hierac_apply0(coeffs, coeffs_list, f):
    """
    using coeffs as a template, apply function to coeffs_list,
    which is a list of coeffs.
    """
    if isinstance(coeffs, np.ndarray):
        return f(coeffs_list)

    r = []
    for i, c in enumerate(coeffs):
        _ = hierac_apply0(c, map(itemgetter(i), coeffs_list), f)
        r.append(_)
    # r = [hierac_apply(c, map(itemgetter(i), coeffs_list))
    #      for i, c in enumerate(coeffs)]

    return r


def hierac_apply(coeffs_list, f=None):
    """
    using coeffs as a template, apply function to coeffs_list,
    which is a list of coeffs. coeffs is set to 0th element.
    """
    if f is None:
        def f(a):
            return np.nanmean(a, axis=0)

    coeffs = coeffs_list[0]

    return hierac_apply0(coeffs, coeffs_list, f)


def get_subimage(arr):
        return np.array(arr)[:, :-1, :-1]


def show_decomposed(original_image, diff_im_wvt, fig=None,
                    imshow_kwargs=None):

    from mpl_toolkits.axes_grid1 import ImageGrid

    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 5))

    nrow = len(diff_im_wvt) + 1  # +1 for original image

    if imshow_kwargs is None:
        imshow_kwargs = dict(interpolation="none")

    ax, = ImageGrid(fig, (1, nrow, 1), (1, 1), axes_pad=0.1)
    ax.imshow(original_image, **imshow_kwargs)

    for irow, imlist in enumerate(diff_im_wvt):
        if irow == 0:
            grid = ImageGrid(fig, (1, nrow, irow + 2), (1, 1), axes_pad=0.1)
            print(imlist.shape)
            grid[0].imshow(imlist, **imshow_kwargs)
        else:
            grid = ImageGrid(fig, (1, nrow, irow + 2), (3, 1), axes_pad=0.1)
            for ax, im in zip(grid, imlist):
                ax.imshow(im, **imshow_kwargs)
