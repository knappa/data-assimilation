from typing import List

import numpy as np
from an_cockrell import AnCockrellModel
from attrs import Factory, define, field


@define(kw_only=True)
class PhenotypeKFAnCockrell:
    num_phenotypes: int = field(init=True)

    # PCA from 2017*41 dimensions to 3 by matrix C (3,2017*41)
    # then each phenotype_i from gaussian with mean m_i (3,) and cov P_i (3,3)
    # -> exp( -(1/2) (Cx-m_i)^T P_i^{-1} (Cx-m_i) )
    pca_matrix: np.ndarray = field(init=True)
    phenotype_weight_means: np.ndarray = field(init=True)
    phenotype_weight_covs: np.ndarray = field(init=True)

    # initial distribution
    phenotype_distribution: np.ndarray = field()
    per_phenotype_means: np.ndarray = field(init=True)
    per_phenotype_covs: np.ndarray = field(init=True)

    # noinspection PyUnresolvedReferences
    @phenotype_distribution.default
    def phenotype_distribution_default(self):
        return np.full(self.num_phenotypes, 1 / self.num_phenotypes, dtype=np.float64)

    ensemble: List[AnCockrellModel] = Factory(list)
    ensemble_projection: np.ndarray = field(init=False)

    # noinspection PyUnresolvedReferences
    @ensemble_projection.default
    def ensemble_projection_default(self):
        return np.zeros(0, dtype=np.float64)

    initial_time : int = -1

    def __attrs_post_init__(self):
        assert (
            self.num_phenotypes
            == self.phenotype_weight_means.shape[0]
            == self.phenotype_weight_covs.shape[0]
            == self.phenotype_distribution.shape[0]
        ), "mismatch in number of phenotypes"
        assert (
            self.phenotype_weight_means.shape[1]
            == self.phenotype_weight_covs.shape[1]
            == self.phenotype_weight_covs.shape[2]
        ), "dimension mismatch"

        # TODO: if the per_phenotype_means/covs is only for 1 phenotype, expand it to all N equally

    def initialize_ensemble(self, *, t: int = 0) -> None:
        """
        Initialize the ensemble.

        :param t: initial time
        :return:
        """
        # TODO: initialization uses different parameters than the running state (Ugh) what to do here?
        pass

    def project_to(self, *, t: int) -> None:
        """
        Project the ensemble to time t.
        :param t: time
        :return: None
        """
        # TODO: decide how much of the ensemble to save (summary/microstate, for how many time steps, etc.)
        pass

    def kf_update(self, *, t: int, types: List[str], measurements: np.ndarray) -> None:
        """

        :param t: time
        :param types: list of measured quantities
        :param measurements: values of measured quantities
        :return:
        """
        # TODO: check the time
        pass
