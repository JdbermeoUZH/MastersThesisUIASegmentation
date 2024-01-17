import torch


# Helper functions for the normal distribution


#
# adapted from https://gist.github.com/ChuaCheowHuan/18977a3e77c0655d945e8af60633e4df?permalink_comment_id=4396301#gistcomment-4396301
#
def kl_divergence(m_to, S_to, m_fr, S_fr):
    """Calculate `KL(to||fr)`, where `m_to` and `m_fr` are means and `S_to` and `S_fr` are covariance matrices"""

    d = m_fr - m_to

    term1 = torch.trace(torch.linalg.solve(S_fr, S_to))
    term2 = torch.linalg.slogdet(S_fr)[0] - torch.linalg.slogdet(S_to)[0]
    term3 = torch.dot(d, torch.linalg.solve(S_fr, d))
    return (term1 + term2 + term3 - len(d))/2.


def fit_distribution(x, diagonal_cov=False):
    mean = x.mean(0)
    if diagonal_cov:
        # Estimating diagonal covariance matrix
        var = x.var(0)
        cov = torch.diag(var)
    else:
        # Estimating full covariance matrix
        cov = x.T.cov()

    return mean, cov
