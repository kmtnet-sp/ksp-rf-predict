import numpy as np
import pywt
import pywvt_helper
from pca_helper import apply_pca
import pickle


# stamp cleaning


def clean(d, lower_cut, fill_value):
    diff_images = d.copy()
    diff_images[diff_images < lower_cut] = fill_value

    return diff_images


def get_cleaned_stamps(stamps):
    stamps_cleaned = clean(stamps, lower_cut=-100, fill_value=0)
    ss = stamps_cleaned

    return ss


def get_center_cut_stamps(stamps):

    # center_cut_images = np.array([reshape(im)[6:-6,6:-6] for im in ss])
    center_cut_images = stamps[:, 7:-7, 7:-7]
    return center_cut_images


def stamp_reshape(im):
    # return np.reshape(im, (21, 21)) # if reshaping is needed
    return im


def get_wvt_coeff(ss):
    def wvt(im):
        return pywt.wavedec2(im, "db1", level=2)

    wvt_diff_images = [wvt(stamp_reshape(im)) for im in ss]

    def get_subimage(arr):
        return np.array(arr)[:, :-1, :-1]

    # convert list of tree of images, to a tree of arrays.
    # coeff_array0 = hierac_apply(wvt_diff_images, f=np.array)
    coeff_array = pywvt_helper.hierac_apply(wvt_diff_images,
                                            f=get_subimage)

    return coeff_array


class WvtPcaFeature(object):
    def __init__(self, pickle_fn):
        self._load_pickle(pickle_fn)

    def _load_pickle(self, fn):
        # print "loading PCA data"
        self._pca_vectors = pickle.load(open(fn))

    def extract(self, stamps):
        ss = get_cleaned_stamps(stamps)
        center_cut_images = get_center_cut_stamps(stamps)
        coeff_array = get_wvt_coeff(ss)

        _ = self._pca_vectors
        pcaC, pca0, pca10, pca11, pca12, pca20, pca21, pca22 = _

        pcC = apply_pca(pcaC, center_cut_images)
        pc0 = apply_pca(pca0, coeff_array[0])
        pc10 = apply_pca(pca10, coeff_array[1][0])
        pc11 = apply_pca(pca11, coeff_array[1][1])
        pc12 = apply_pca(pca12, coeff_array[1][2])
        pc20 = apply_pca(pca20, coeff_array[2][0])
        pc21 = apply_pca(pca21, coeff_array[2][1])
        pc22 = apply_pca(pca22, coeff_array[2][2])

        features = np.hstack([pcC, pc0, pc10, pc11, pc12, pc20, pc21, pc22])

        return features


if __name__ == "__main__":
    feature_root = "ttt"
    outdir = "."
    pickle_name = "%s.wvt_pca_vectors.pkl" % (feature_root,)

    import os
    fn = os.path.join(outdir, pickle_name)

    wvt_pca_feature = WvtPcaFeature(fn)
    stamps = None
    features = wvt_pca_feature.extract(stamps)
    features
