from dataclasses import dataclass
from typing import override

import dwave.samplers
from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo


@dataclass
class LUNASA(Core):
    """
    A module for solving a qubo problem using simulated annealing

    :param num_reads: The number of reads to perform
    """

    num_reads: int = 100


    @override
    def preprocess(self, data: Qubo) -> Result:
        device = dwave.samplers.SimulatedAnnealingSampler()
        self._result = device.sample_qubo(data.as_dict(), num_reads=self.num_reads)
        return Data(None)

    @override
    def postprocess(self, data: Data) -> Result:
        return Data(Other(self._result.lowest().first.sample))
