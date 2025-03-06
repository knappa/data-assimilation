from typing import List, Optional, Callable

import an_cockrell
from an_cockrell import AnCockrellModel
import attrs
from attrs import define, field, Factory

@define
class PhenotypeKFAnCockrell():

    phenotype_weight_function : List[Callable] = field(init=True)

    ensemble: List[AnCockrellModel] = Factory(list)

    pass
