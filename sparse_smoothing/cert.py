import numpy as np
import gmpy2
from tqdm.autonotebook import tqdm
from statsmodels.stats.proportion import proportion_confint
from itertools import product
from collections import defaultdict


def regions_binary(ra, rd, pf_plus, pf_minus, precision=1000):
    """
    Construct (px, px_tilde, px/px_tilde) regions used to find the certified radius for binary data.

    Intuitively, pf_minus controls rd and pf_plus controls ra.

    Parameters
    ----------
    ra: int
        Number of ones y has added to x
    rd : int
        Number of ones y has deleted from x
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    precision: int
        Numerical precision for floating point calculations

    Returns
    -------
    regions: array-like, [None, 3]
        Regions of constant probability under px and px_tilde,
    """

    pf_plus, pf_minus = gmpy2.mpfr(pf_plus), gmpy2.mpfr(pf_minus)
    with gmpy2.context(precision=precision):
        if pf_plus == 0:
            px = pf_minus ** rd
            px_tilde = pf_minus ** ra

            return np.array([[1 - px, 0, float('inf')],
                             [px, px_tilde, px / px_tilde],
                             [0, 1 - px_tilde, 0]
                             ])

        if pf_minus == 0:
            px = pf_plus ** ra
            px_tilde = pf_plus ** rd
            return np.array([[1 - px, 0, float('inf')],
                             [px, px_tilde, px / px_tilde],
                             [0, 1 - px_tilde, 0],
                             ])
        max_q = ra + rd
        i_vec = np.arange(0, max_q + 1)

        T = ra * ((pf_plus / (1 - pf_plus)) ** i_vec) + \
            rd * ((pf_minus / (1 - pf_minus)) ** i_vec)

        ratio = np.zeros_like(T)
        px = np.zeros_like(T)
        px[0] = 1

        for q in range(0, max_q + 1):
            ratio[q] = (pf_plus/(1-pf_minus)) ** (q - rd) * \
                (pf_minus/(1-pf_plus)) ** (q - ra)

            if q == 0:
                continue

            for i in range(1, q + 1):
                px[q] = px[q] + ((-1) ** (i + 1)) * T[i] * px[q - i]
            px[q] = px[q] / q

        scale = ((1-pf_plus) ** ra) * ((1-pf_minus) ** rd)

        px = px * scale

        regions = np.column_stack((px, px / ratio, ratio))
        if pf_plus+pf_minus > 1:
            # reverse the order to maintain decreasing sorting
            regions = regions[::-1]
        return regions


def triplets(r, k):
    """
    Generate all triplets of positive integers that add up to r.
    Parameters
    ----------
    r: int
        The sum of the triplets (corresponding to the radius).
    k: int
        Number of discrete categories.

    Returns
    -------
    triplets_r: list(tuple)
        A list of triplets that sum to to r.
    """

    triplets_list = []
    for q in range(0, r+1):
        for p in range(0 if k > 2 else r-q, r+1-q):
            triplets_list.append((q, p, r-q-p))
    return triplets_list


def regions_discrete(ra, rd, rc, k, pf_plus, pf_minus, precision=1000):
    """
    Construct (px, px_tilde, px/px_tilde) regions used to find the certified radius for general discrete data.

    Note: if pf_plus = pf_minus any combination of ra+rd+rc=r gives the same result.

    ra: int
        Number of zeros changed to non-zeros from x to x_tilde.
    rd : int
        Number of non-zeros changed to zeros from x to x_tilde.
    rc : int
        Number of non-zeros changed to other non-zero values from x to x_tilde.
    k: int
        Number of discrete categories.
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a non-zero.
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a non-zero to a zero.
    precision: int
        Numerical precision for floating point calculations.
    """
    with gmpy2.context(precision=precision):
        pf_plus, pf_minus = gmpy2.mpfr(pf_plus), gmpy2.mpfr(pf_minus)
        one =  gmpy2.mpfr(1)

    ra, rd, rc = int(ra), int(rd), int(rc)

    a0 = (one-pf_plus)
    b0 = pf_plus/(k-1)
    c0 = one - a0 - b0
    dist_0 = [a0, b0, c0]

    a1 = (one-pf_minus)
    b1 = pf_minus/(k-1)
    c1 = one - a1 - b1
    dist_1 = [a1, b1, c1]

    regions = defaultdict(float)
    for triplet in product(triplets(ra, k),
                           triplets(rd, k),
                           triplets(rc, k)):
        q0, p0, m0 = triplet[0]
        q1, p1, m1 = triplet[1]
        q2, p2, m2 = triplet[2]

        ratio = 1
        ratio *= (a0/b1) ** (q0-p1)
        ratio *= (b0/a1) ** (p0-q1)
        ratio *= (c0/c1) ** (m0-m1)
        ratio *= (a1/b1) ** (q2-p2)

        # compute the product of three multinomial distributions
        px = 1
        for (qi, pi, mi), ri, dist in zip(triplet,
                                          [ra, rd, rc],
                                          [dist_0, dist_1, dist_1]):
            px *= dist[0] ** qi
            px *= dist[1] ** pi
            px *= dist[2] ** mi
            px *= gmpy2.fac(ri) / gmpy2.fac(qi) / gmpy2.fac(pi) / gmpy2.fac(mi)

        regions[float(ratio)] += px

    regions = np.array(list(regions.items()))
    srt = regions[:, 0].argsort()[::-1]

    return np.column_stack((regions[srt, 1], regions[srt, 1]/regions[srt, 0], regions[srt, 0]))


def joint_regions(ra_adj, rd_adj, ra_att, rd_att,
                  pf_plus_adj, pf_minus_adj, pf_plus_att, pf_minus_att,
                  regions_adj=None, regions_att=None):
    """
    Construct regions for certifying two separate sub-spaces at once.
    Form the product of regions obtained with `regions_binary`.

    Parameters
    ----------
    ra_adj: int
        Number of ones y has added to the adjacency matrix
    rd_adj : int
        Number of ones y has deleted from adjacency matrix
    ra_att: int
        Number of ones y has added to the attribute matrix
    rd_att : int
        Number of ones y has deleted from attribute matrix
    pf_plus_adj : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one for the adjacency matrix.
    pf_minus_adj: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero  for the adjacency matrix.
    pf_plus_att : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one for the attribute matrix.
    pf_minus_att: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero  for the attribute matrix.
    regions_adj : array-like, [None, 3]
        Regions of constant probability under p_x and p_y for ADJ,
    regions_att : array-like, [None, 3]
        Regions of constant probability under p_x and p_y for ATT,
    Returns
    -------
    regions: array-like, [None, 3]
        Regions of constant probability under p_x and p_y,
    """
    if regions_adj is None:
        regions_adj = regions_binary(
            ra=ra_adj, rd=rd_adj, pf_plus=pf_plus_adj, pf_minus=pf_minus_adj)
    if regions_att is None:
        regions_att = regions_binary(
            ra=ra_att, rd=rd_att, pf_plus=pf_plus_att, pf_minus=pf_minus_att)

    cross_regions = []
    for px_adj, py_adj, ratio_adj in regions_adj:
        for px_att, py_att, ratio_att in regions_att:
            px = px_adj * px_att
            py = py_adj * py_att
            cross_regions.append((px, py, ratio_adj * ratio_att))

    return regions_adj, regions_att, np.array(cross_regions)


def compute_rho(regions, p_emp, verbose=False, is_sorted=True, reverse=False):
    """
    Compute the worst-case probability of the adversary.
    For the binary-class certificate if rho>0.5 the instance is certifiable robust.

    Parameters
    ----------
    regions : array-like, [?, 3]
        Regions of constant probability under p_x and p_y,
    p_emp : float
        Empirical probability of the majority class
    verbose : bool
        Verbosity
    is_sorted : bool
        Whether the regions are sorted, e.g. regions from `regions_binary` are automatically sorted.
    reverse : bool
        Whether to consider the sorting in reverse order which we need for computing an upper_bound.

    Returns
    -------
    p_adver : float
        The worst-case probability of the adversary.
    """
    if is_sorted:
        sorted_regions = regions
    else:
        # sort in descending order
        sorted_regions = sorted(
            list(regions), key=lambda a: a[2], reverse=True)
    sorted_regions = reversed(sorted_regions) if reverse else sorted_regions

    if verbose:
        region_sum = sum(map(lambda x: x[0], regions))
        print('region_sum_is', region_sum)

    acc_p_clean = 0.0
    acc_p_adver = 0.0

    for i, (p_clean, p_adver, _) in enumerate(sorted_regions):
        # break early so the sums only reflect up to H*-1
        if acc_p_clean + p_clean >= p_emp:
            break
        if p_clean > 0:
            acc_p_clean += p_clean
            acc_p_adver += p_adver

    rho = acc_p_adver

    if verbose:
        print('clean', float(acc_p_clean), 'adver', float(
            acc_p_adver), 'counter={}/{}'.format(i, len(regions)))

    # there is some probability left
    if p_emp - acc_p_clean > 0 and i < len(regions):
        addition = (p_emp - acc_p_clean) * (p_adver / p_clean)
        rho += addition

        if verbose:
            print('ratio', float(p_adver / p_clean), 'diff',
                  float(p_emp - acc_p_clean), 'addition', float(addition))
            print(float(p_adver), float(p_clean))
            print(rho > acc_p_adver)

    return rho


def compute_rho_for_many(regions, p_emps, is_sorted=True, reverse=False):
    """
    Compute the worst-case probability of the adversary for many p_emps at once.

    Parameters
    ----------
    regions : array-like, [?, 3]
        Regions of constant probability under p_x and p_y,
    p_emps : array-like [?]
        Empirical probabilities per node.
    is_sorted : bool
        Whether the regions are sorted, e.g. regions from `regions_binary` are automatically sorted.
    reverse : bool
        Whether to consider the sorting in reverse order.

    Returns
    -------
    p_adver : array-like [?]
        The worst-case probability of the adversary.
    """
    sort_direction = -1 if reverse else 1
    if not is_sorted:
        o = regions[:, 2].argsort()[::-sort_direction]
        regions = regions[o]
    else:
        regions = regions[::sort_direction]

    # add one empty region to have easier indexing
    regions = np.row_stack(([0, 0, 0], regions))

    cumsum = np.cumsum(regions[:, :2], 0)
    h_stars = (cumsum[:, 0][:, None] >= p_emps).argmax(0)
    h_stars[h_stars > 0] -= 1

    h_star_cumsums = cumsum[h_stars]

    acc_p_clean = h_star_cumsums[:, 0]
    acc_p_adver = h_star_cumsums[:, 1]

    # add the missing probability for those that need it
    flt = (p_emps - acc_p_clean > 0) & (h_stars + 1 < len(regions))
    addition = (p_emps[flt] - acc_p_clean[flt]) * \
        regions[h_stars[flt] + 1, 1] / regions[h_stars[flt] + 1, 0]
    acc_p_adver[flt] += addition

    acc_p_adver[h_stars == -1] = 0

    return acc_p_adver.astype('float')


def p_lower_from_votes(votes, pre_votes, alpha, n_samples):
    """
    Estimate a lower bound on the probability of the majority class using a Binomial confidence interval.

    Parameters
    ----------
    votes: array_like [n_nodes, n_classes]
        Votes per class for each sample
    pre_votes: array_like [n_nodes, n_classes]
        Votes (based on fewer samples) to determine the majority (and the second best) class
    alpha : float
        Significance level
    n_samples : int
        Number of MC samples
    Returns
    -------
    p_lower: array-like [n_nodes]
        Lower bound on the probability of the majority class

    """
    # Multiple by 2 since we are only need a single side
    n_best = votes[np.arange(votes.shape[0]), pre_votes.argmax(1)]
    p_lower = proportion_confint(
        n_best, n_samples, alpha=2 * alpha, method="beta")[0]
    return p_lower


def p_lower_upper_from_votes(votes, pre_votes, conf_alpha, n_samples):
    """
    Estimate a lower bound on the probability of the majority class and an upper bound on the probability
    of the second best class using c Binomial confidence intervals and Bonferroni correction.

    Parameters
    ----------
    votes: array_like [n_nodes, n_classes]
        Votes per class for each sample
    pre_votes: array_like [n_nodes, n_classes]
        Votes (based on fewer samples) to determine the majority (and the second best) class
    conf_alpha : float
        Significance level
    n_samples : int
        Number of MC samples

    Returns
    -------
    p_lower: array-like [n_nodes]
        Lower bound on the probability of the majority class
    p_upper: array-like [n_nodes]
        Upper bound on the probability of the second best class
    """
    n, nc = votes.shape

    pre_votes_max = pre_votes.argsort()[:, -2:]

    n_second_best = votes[np.arange(n), pre_votes_max[:, 0]]
    n_best = votes[np.arange(n), pre_votes_max[:, 1]]

    # Bonferroni implies we should divide by the number of classes nc
    p_lower = proportion_confint(
        n_best, n_samples, alpha=2 * conf_alpha / nc, method="beta")[0]
    p_upper = proportion_confint(
        n_second_best, n_samples, alpha=2 * conf_alpha / nc, method="beta")[1]

    return p_lower, p_upper


def max_radius_for_p_emp(pf_plus, pf_minus, p_emp, which, upper=100, verbose=False):
    """
    Find the maximum radius we can certify individually (either ra or rd) using bisection.

    Parameters
    ----------
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    p_emp : float
        Empirical probability of the majority class
    which : string
        'ra': find max_{ra, rd=0}
        'rd': find max_{rd, ra=0}
    upper : int
        An upper bound on the maximum radius
    verbose : bool
        Verbosity.

    Returns
    -------
    max_r : int
        The maximum certified radius s.t. the probability of the adversary is above 0.5.

    """
    initial_upper = upper
    lower = 1
    r = 1

    while lower < upper:
        r = lower + (upper - lower) // 2
        if which == 'ra':
            ra = r
            rd = 0
        elif which == 'rd':
            ra = 0
            rd = r
        else:
            raise ValueError('which can only be "ra" or "rd"')

        cur_rho = compute_rho(regions_binary(
            ra=ra, rd=rd, pf_plus=pf_plus, pf_minus=pf_minus), p_emp)
        if verbose:
            print(r, float(cur_rho))

        if cur_rho > 0.5:
            if lower == r:
                break
            lower = r
        else:
            upper = r

    if r == initial_upper or r == initial_upper - 1:
        if verbose:
            print('r = upper, restart the search with a larger upper bound')
        return max_radius_for_p_emp(pf_plus=pf_plus, pf_minus=pf_minus,
                                    p_emp=p_emp, which=which, upper=2*upper, verbose=verbose)

    return r


def min_p_emp_for_radius_1(pf_plus, pf_minus, which, lower=0.5, verbose=False):
    """
    Find the smallest p_emp for which we can certify a radius of 1 using bisection.


    Parameters
    ----------
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    which : string
        'ra': find min_{p_emp, ra=1, rd=0}
        'rd': find min_{p_emp, rd=1, ra=0}
    lower : float
        A lower bound on the minimum p_emp.
    verbose : bool
        Verbosity.

    Returns
    -------
    min_p_emp : float
        The minimum p_emp.
    """
    initial_lower = lower
    upper = 1
    p_emp = 0

    if which == 'ra':
        ra = 1
        rd = 0
    elif which == 'rd':
        ra = 0
        rd = 1
    else:
        raise ValueError('which can only be "ra" or "rd"')

    while lower < upper:
        p_emp = lower + (upper - lower) / 2

        cur_rho = compute_rho(regions_binary(
            ra=ra, rd=rd, pf_plus=pf_plus, pf_minus=pf_minus), p_emp)
        if verbose:
            print(p_emp, float(cur_rho))

        if cur_rho < 0.5:
            if lower == p_emp:
                break
            lower = p_emp
        elif abs(cur_rho - 0.5) < 1e-10:
            break
        else:
            upper = p_emp

    if p_emp <= initial_lower:
        if verbose:
            print(
                'p_emp <= initial_lower, restarting the search with a smaller lower bound')
        return min_p_emp_for_radius_1(
            pf_plus=pf_plus, pf_minus=pf_minus, which=which, lower=lower*0.5, verbose=verbose)

    return p_emp


def binary_certificate_grid(pf_plus, pf_minus, p_emps, reverse=False, regions=None, max_ra=None, max_rd=None, progress_bar=True):
    """
    Compute rho for all given p_emps and for all combinations of radii up to the maximum radii.

    Parameters
    ----------
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    p_emps : array-like [n_nodes]
        Empirical probabilities per node.
    reverse : bool
        Whether to consider the sorting in reverse order.
    regions : dict
        A pre-computed dictionary of regions
    progress_bar : bool
        Whether to show a tqdm progress bar

    Returns
    -------
    radii : array-like, [n_nodes, max_ra, max_rd]
        Probabilities of the adversary. Node is certified if [:, :, :]>0.5
    regions : dict
        A pre-computed dictionary of regions
    max_ra : int
        Maximum certified addition radius
    max_rd : int
        Maximum certified deletion radius
    """
    if progress_bar:
        def bar(loop):
            return tqdm(loop)
    else:
        def bar(loop):
            return loop

    if regions is None:
        # compute the maximum possible ra and rd we can certify for the largest p_emp
        if max_ra is None or max_rd is None:
            max_p_emp = p_emps.max()
            max_ra = max_radius_for_p_emp(
                pf_plus=pf_plus, pf_minus=pf_minus, p_emp=max_p_emp, which='ra', upper=100)
            max_rd = max_radius_for_p_emp(
                pf_plus=pf_plus, pf_minus=pf_minus, p_emp=max_p_emp, which='rd', upper=100)
            min_p_emp = min(min_p_emp_for_radius_1(pf_plus, pf_minus, 'ra'),
                            min_p_emp_for_radius_1(pf_plus, pf_minus, 'rd'))

            print(f'max_ra={max_ra}, max_rd={max_rd}, min_p_emp={min_p_emp:.4f}')

        regions = {}
        for ra in bar(range(max_ra + 2)):
            for rd in range(max_rd + 2):
                regions[(ra, rd)] = regions_binary(
                    ra=ra, rd=rd, pf_plus=pf_plus, pf_minus=pf_minus)

    n_nodes = len(p_emps)
    arng = np.arange(n_nodes)
    radii = np.zeros((n_nodes, max_ra + 2, max_rd + 2))

    for (ra, rd), regions_ra_rd in bar(regions.items()):
        if ra + rd == 0:
            radii[arng, ra, rd] = 1
        else:
            radii[arng, ra, rd] = compute_rho_for_many(
                regions=regions_ra_rd, p_emps=p_emps, is_sorted=True, reverse=reverse)

    return radii, regions, max_ra, max_rd


def joint_binary_certificate_grid(pf_plus_adj, pf_minus_adj, pf_plus_att, pf_minus_att, p_emps, reverse=False,
                                  cross_regions=None, progress_bar=True):
    """
    Compute rho for all given p_emps and for all combinations of radii up to the maximum radii.

    Parameters
    ----------
    pf_plus_adj : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one for the adjacency matrix.
    pf_minus_adj: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero  for the adjacency matrix.
    pf_plus_att : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one for the attribute matrix.
    pf_minus_att: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero  for the attribute matrix.
    p_emps : array-like [n_nodes]
        Empirical probabilities per node.
    reverse : bool
        Whether to consider the sorting in reverse order.
    cross_regions:
    progress_bar : bool
        Whether to show a tqdm progress bar

    Returns
    -------

    """
    if progress_bar:
        def bar(loop):
            return tqdm(loop)
    else:
        def bar(loop):
            return loop

    max_p_emp = p_emps.max()

    if cross_regions is None:
        max_ra_adj = max_radius_for_p_emp(
            pf_plus=pf_plus_adj, pf_minus=pf_minus_adj, p_emp=max_p_emp, which='ra')
        max_rd_adj = max_radius_for_p_emp(
            pf_plus=pf_plus_adj, pf_minus=pf_minus_adj, p_emp=max_p_emp, which='rd')

        max_ra_att = max_radius_for_p_emp(
            pf_plus=pf_plus_att, pf_minus=pf_minus_att, p_emp=max_p_emp, which='ra')
        max_rd_att = max_radius_for_p_emp(
            pf_plus=pf_plus_att, pf_minus=pf_minus_att, p_emp=max_p_emp, which='rd')

        print(f'max_ra_adj={max_ra_adj}, max_rd_adj={max_rd_adj}, max_ra_att={max_ra_att}, max_rd_att={max_rd_att}')

        extra = 2
        regions_adj = {}
        regions_att = {}
        # precomputed individual regions for the adj
        for ra_adj, rd_adj in bar(product(range(max_ra_adj + extra), range(max_rd_adj + extra))):
            regions_adj[ra_adj, rd_adj] = regions_binary(ra=ra_adj, rd=rd_adj, pf_plus=pf_plus_adj,
                                                         pf_minus=pf_minus_adj)

        # precomputed individual regions for the att
        for ra_att, rd_att in bar(product(range(max_ra_att + extra), range(max_rd_att + extra))):
            regions_att[ra_att, rd_att] = regions_binary(ra=ra_att, rd=rd_att, pf_plus=pf_plus_att,
                                                         pf_minus=pf_minus_att)

        # prepare all the regions for the grid
        cross_regions = {}
        for ra_adj, rd_adj, ra_att, rd_att in bar(product(range(max_ra_adj + extra),
                                                          range(max_rd_adj + extra),
                                                          range(max_ra_att + extra),
                                                          range(max_rd_att + extra))):
            _, _, cur_cross_regions = joint_regions(
                ra_adj=ra_adj, rd_adj=rd_adj, ra_att=ra_att, rd_att=rd_att,
                pf_plus_adj=pf_plus_adj, pf_minus_adj=pf_minus_adj,
                pf_plus_att=pf_plus_att, pf_minus_att=pf_minus_att,
                regions_adj=regions_adj[ra_adj, rd_adj],
                regions_att=regions_att[ra_att, rd_att]
            )

            cross_regions[ra_adj, rd_adj, ra_att, rd_att] = cur_cross_regions

    max_ra_adj, max_rd_adj, max_ra_att, max_rd_att = np.array(
        list(cross_regions.keys())).max(0) + 1

    n_nodes = len(p_emps)
    arng = np.arange(n_nodes)
    heatmap = np.zeros(
        (n_nodes, max_ra_adj, max_rd_adj, max_ra_att, max_rd_att))

    # compute the radius
    for ra_adj, rd_adj, ra_att, rd_att in cross_regions:
        if ra_adj + rd_adj + ra_att + rd_att == 0:
            heatmap[arng, ra_adj, rd_adj, ra_att, rd_att] = 1
        else:
            rad = compute_rho_for_many(regions=cross_regions[ra_adj, rd_adj, ra_att, rd_att],
                                       p_emps=p_emps, is_sorted=True, reverse=reverse)
            heatmap[arng, ra_adj, rd_adj, ra_att, rd_att] = rad

    return heatmap, cross_regions


def binary_certificate(votes, pre_votes, n_samples, conf_alpha, pf_plus, pf_minus):
    """
    Compute both the binary-class certificate 2D grid (for all pairs of ra and rd)
    where grid_base > 0.5 means the instance is robust, and the multi-class
    certificate where grid_lower > grid_upper means the instance is robust.

    Parameters
    ----------
    votes: array_like [n_nodes, n_classes]
        Votes per class for each sample
    pre_votes: array_like [n_nodes, n_classes]
        Votes (based on fewer samples) to determine the majority (and the second best) class
    n_samples : int
        Number of MC samples
    conf_alpha : float
        Significance level
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero

    Returns
    -------

    """
    # binary-class certificate
    p_emps = p_lower_from_votes(
        votes=votes, pre_votes=pre_votes, alpha=conf_alpha, n_samples=n_samples)
    grid_base, regions, base_max_ra, base_max_rd = binary_certificate_grid(pf_plus=pf_plus, pf_minus=pf_minus,
                                                                                   p_emps=p_emps, reverse=False, progress_bar=True)

    # multi-class certificate
    p_lower, p_upper = p_lower_upper_from_votes(
        votes=votes, pre_votes=pre_votes, conf_alpha=conf_alpha, n_samples=n_samples)
    grid_lower, *_ = binary_certificate_grid(pf_plus=pf_plus, pf_minus=pf_minus, p_emps=p_lower, reverse=False,
                                                         regions=regions, max_ra=base_max_ra, max_rd=base_max_rd, progress_bar=True)

    grid_upper, *_, = binary_certificate_grid(pf_plus=pf_plus, pf_minus=pf_minus, p_emps=p_upper, reverse=True,
                                                          regions=regions, max_ra=base_max_ra, max_rd=base_max_rd, progress_bar=True,)

    return grid_base, grid_lower, grid_upper


def joint_binary_certificate(votes, pre_votes, n_samples, conf_alpha,
                            pf_plus_adj, pf_minus_adj, pf_plus_att, pf_minus_att
                            ):
    """
    Compute both the binary-class certificate 4D grid (all combinations of ra and rd)
    where grid_base > 0.5 means the instance is robust, and the multi-class
    certificate where grid_lower > grid_upper means the instance is robust.

    Parameters
    ----------
    votes: array_like [n_nodes, n_classes]
        Votes per class for each sample
    pre_votes: array_like [n_nodes, n_classes]
        Votes (based on fewer samples) to determine the majority (and the second best) class
    n_samples : int
        Number of MC samples
    conf_alpha : float
        Significance level
    pf_plus_adj : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one for the adjacency matrix.
    pf_minus_adj: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero  for the adjacency matrix.
    pf_plus_att : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one for the attribute matrix.
    pf_minus_att: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero  for the attribute matrix.

    Returns
    -------

    """
    # binary-class certificate
    p_emps = p_lower_from_votes(
        votes=votes, pre_votes=pre_votes, alpha=conf_alpha, n_samples=n_samples)
    grid_base, _ = joint_binary_certificate_grid(
        pf_plus_adj=pf_plus_adj, pf_minus_adj=pf_minus_adj,
        pf_plus_att=pf_plus_att, pf_minus_att=pf_minus_att,
        p_emps=p_emps, reverse=False, progress_bar=True)

    # multi-class certificate
    p_lower, p_upper = p_lower_upper_from_votes(
        votes=votes, pre_votes=pre_votes, conf_alpha=conf_alpha, n_samples=n_samples)

    grid_lower, cross_regions = joint_binary_certificate_grid(
        pf_plus_adj=pf_plus_adj, pf_minus_adj=pf_minus_adj,
        pf_plus_att=pf_plus_att, pf_minus_att=pf_minus_att,
        p_emps=p_lower, reverse=False, progress_bar=True)

    grid_upper, _ = joint_binary_certificate_grid(
        pf_plus_adj=pf_plus_adj, pf_minus_adj=pf_minus_adj,
        pf_plus_att=pf_plus_att, pf_minus_att=pf_minus_att,
        p_emps=p_upper, reverse=True, cross_regions=cross_regions, progress_bar=True)

    return grid_base, grid_lower, grid_upper
