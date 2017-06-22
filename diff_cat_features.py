import numpy as np


ks = [u'CLASS_STAR', u'ELONGATION',
      u'ERRAWIN_IMAGE', u'ERRBWIN_IMAGE', u'ERRTHETAWIN_IMAGE', u'FLAGS',
      u'FLUXERR_APER', u'FLUXERR_AUTO',
      u'FLUXERR_ISO', u'FLUX_APER', u'FLUX_AUTO', u'FLUX_MAX', u'FLUX_RADIUS',
      u'FWHM_IMAGE', u'KSP_SOURCE_CLASS', u'SNR_WIN']


def source_class_to_value(column):
    flag_values = 2**np.arange(len(column[0]))
    # _v = np.array(list(diff_cat_df["KSP_SOURCE_CLASS"].values))

    return (flag_values * column).sum(axis=1)


def get_diff_cat_features(diff_cat):

    vv = []
    for k in ks:
        if k == u'KSP_SOURCE_CLASS':
            v = source_class_to_value(diff_cat[k])
            vv.append(v)
        else:
            vv.append(diff_cat[k])

    xx2 = np.array(vv).T

    return xx2


if __name__ == "__main__":
    diff_cat = None
    diff_cat_features = get_diff_cat_features(diff_cat)
