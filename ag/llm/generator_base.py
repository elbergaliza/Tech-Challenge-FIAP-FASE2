from abc import ABC, abstractmethod
from .laudos_type import EntradaLaudo


class LaudoGenerator(ABC):
    @abstractmethod
    def gerar(self, entrada: EntradaLaudo) -> str:
        raise NotImplementedError