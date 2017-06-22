import numpy as np
from sklearn.externals import joblib


class RFPredictor(object):

    PROB_TRESH = 0.54

    def __init__(self, pkl_name):

        self.clf = joblib.load(pkl_name)

    def predict_proba(self, features):
        predicted = self.clf.predict_proba(features)
        return predicted


def get_config():

    feature_root = "ttt"
    outdir = "."
    pickle_name = "%s.wvt_pca_vectors.pkl" % (feature_root,)

    import os
    wvt_pca_vector_pkl_name = os.path.join(outdir, pickle_name)

    tmpl = "true_bogus_RF_model.model_%02d.20170602.joblib.pkl"
    sample_i = 7

    rf_model_pkl_name = tmpl % (sample_i,)

    return dict(rf_model_pkl_name=rf_model_pkl_name,
                wvt_pca_vector_pkl_name=wvt_pca_vector_pkl_name)


if __name__ == "__main__":

    import os
    import astropy.io.fits as pyfits

    from diff_cat_features import get_diff_cat_features
    from wvt_pca_features import WvtPcaFeature
    conf = get_config()

    predictor = RFPredictor(conf["rf_model_pkl_name"])
    wvt_pca_feature = WvtPcaFeature(conf["wvt_pca_vector_pkl_name"])

    # load diff_cat and extract stamps
    diff_cat_dir = "PROCESSED/E489/E489-1/Q3/B_Filter/Subtraction"
    fn_basename = ("E489-1.Q3.B.161118_0722.C.037891.061439N2443.0060"
                   ".nh.REF-SUB.cat")

    f = pyfits.open(os.path.join(diff_cat_dir, fn_basename))

    diff_cat = f[2].data
    stamps = diff_cat["VIGNET"]

    features1 = get_diff_cat_features(diff_cat)
    features2 = wvt_pca_feature.extract(stamps)

    features = np.hstack([features1, features2])

    predicted = predictor.predict_proba(features)

    msk1 = (predicted[:, 1] > 0.54)
    # & (features1[:, -2] == 0)
    # msk2 = (predicted[:, 1] > 0.7)
    msk = msk1 # | msk2
    indices = np.arange(len(diff_cat))

    print(msk1.sum())

    diff_cat_selected = diff_cat[msk].copy()
    hdu = pyfits.BinTableHDU(data=diff_cat_selected)
    hdu.writeto("t.cat")

if 0:
    ss = []
    fmt = "fk5; circle(%.5f, %.5f, 10\") # text={%d, %.2f}\n"
    for ra, dec, p, i in zip(diff_cat[msk]["X_WORLD"],
                             diff_cat[msk]["Y_WORLD"],
                             predicted[:, 1][msk],
                             indices[msk]):
        s = fmt % (ra, dec, i, p)
        ss.append(s)
    open("test2.reg", "w").writelines(ss)
