from scipy.linalg import eigh
import networkx


def get_eigenvalue_residuals(graph1, graph2):
    """
    Eigenvector "similarity" of two network graphs (actually the sum of squared differences between
    eigenvalues)
    :param graph1: networkx.graph object to compare
    :param graph2: networkx.graph object to compare
    :return: 0 to infinity (0 is most similar)
    """

    laplacian1 = laplacian_spectrum(graph1, weight="weight")
    laplacian2 = laplacian_spectrum(graph2, weight="weight")

    # select K eigenvalues from the laplacians
    k1 = get_k_eigenvalues(laplacian1)
    k2 = get_k_eigenvalues(laplacian2)
    k = min(k1, k2)  # get smallest k

    # sum of the squared differences between the largest k eigenvalues between the graphs
    return sum((laplacian1[:k] - laplacian2[:k]) ** 2)


def laplacian_spectrum(g, weight='weight'):
    return eigvalsh(networkx.laplacian_matrix(g, weight=weight).todense())


def eigvalsh(a, b=None, lower=True, overwrite_a=True,
             overwrite_b=True, turbo=True, eigvals=None, type=1,
             check_finite=False):
    return eigh(a, b=b, lower=lower, eigvals_only=True,
                overwrite_a=overwrite_a, overwrite_b=overwrite_b,
                turbo=turbo, eigvals=eigvals, type=type,
                check_finite=check_finite)


def get_k_eigenvalues(laplace, minimum_energy=0.9):
    """
    find the smallest k of eigenvalues from a laplace such that the sum of the k largest eigenvalues
    constitutes at least a minimum_energy of the sum of all of the eigenvalues
    :param laplace: laplacian matrix
    :param minimum_energy: the minimum proportion of the sum of all of the eigenvalues that the k of eignevalues
    must meet
    :return: smallest k of eigenvalues that meet the minimum energy requirement
    """
    running_total = 0.0
    total = sum(laplace)
    if total == 0.0:
        return len(laplace)
    for i in range(len(laplace)):
        running_total += laplace[i]
        if running_total / total >= minimum_energy:
            # we have enough
            return i + 1
    return len(laplace)
