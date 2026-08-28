"""

Routines with dealing with the grain size profiles and lithology table

"""

import numpy as np


def get_representative_grain_sizes(cfds: list[list]):
    """Return the representative grain sizes

    Parameters
    ----------
    cfds: list[list]
        len of cfds if number of bins + 1, each element has number of
        profiles + 1 elements.  Suppose there are n bins with boundaries
        [b0, b1), [b1, b2), ... [bn-1, bn)
        and m profiles then we have something like:

        b0  cumfreq10 cumfreq20 ... cumfreqm0
        b1  cumfreq11 cumfreq21 ... cumfreqm1

        bn  cumfreq1n cumfreq2n ... cumfreqmn


    Returns
    -------
    np.array
        The representative sizes of the grain sizes using 2^( (log(bot) + log(top))/2 )

        2^( (log(b0) + log(b1))/2 )
        2^( (log(b1) + log(b2))/2 )

        2^( (log(bn-1) + log(bn))/2 )

    """
    # get first column, the sizes from cfds.
    p = [i[0] for i in cfds]
    return np.array(
        [2 ** ((np.log(bot) + np.log(top)) / 2) for bot, top in zip(p[:-1], p[1:])]
    )


def get_grain_props(pid: int, cfds: list[list], lithtab: list[list]):
    """Return the fractions ie proportions for given profile index.

    Parameters
    ----------
    pid: int
        The grain profile 0-based index

    cfds: list[list]
        See get_represenative_grain_sizes

    lithtab: list[list]
        nbins*nliths x number of grain profiles.  The pid'th column gives the
        info for the pid grain profile.  The column has nbin groups, each group
        has a row for each lithology group.  The actual values give percentages
        of how much lithology is in this bin.  Eg suppose 2 bins, 3 lith
        groups, and the cfds column for our profile is

            0
            10
            100

        then we have 0.1 in bin 1 and 0.9 in bin 2, and suppose our lith column
        is:

            b1  l1      100
            b1  l2        0
            b1  l3        0
            b2  l1       50
            b2  l2        0
            b2  l3       50

        So this grain profile would be
            0.1  0 0
            0.45 0 0.45

    Returns
    -------
    np.array:
        2d array.  nbins x nlith
        A row for each bin, and each row is nlith in length
        and gives the fraction or proportion of grains of this representative
        size.
    """
    # this is the cummulative frequency for this profile
    cfd = [i[pid] for i in cfds]

    # convert to a proportion fraction
    total = cfd[-1]
    prop = [cfd[0] / total, *[(b - a) / total for a, b in zip(cfd[:-1], cfd[1:])]]
    nbins = len(prop)

    # get the number of lith
    nlith = int(len(lithtab) / nbins)
    assert nlith * nbins == len(lithtab)

    if nlith == 1:
        return np.array(prop)[:, None]

    # need the pid'th column of lithfractions
    lfracs = [lf[pid] for lf in lithtab]

    # number of bins x nlith
    weights = np.array(lfracs).reshape(-1, nlith)
    weights = weights / weights.sum(axis=1, keepdims=1)

    return np.expand_dims(prop, 1) * weights
