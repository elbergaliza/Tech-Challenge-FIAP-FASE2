from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ResultadoModelo:
    classe_predita: int                 
    probabilidade_positiva: Optional[float]  
    limiar_decisao: float              


@dataclass(frozen=True)
class ContextoModelo:
    nome_modelo: str                   
    target_name: str                   
    roc_auc_global: Optional[float]     
    acuracia_teste: Optional[float]    
    metadata: Dict[str, Any]            


@dataclass(frozen=True)
class EntradaLaudo:
    resultado: ResultadoModelo
    contexto: ContextoModelo
    resumo_exame: Dict[str, Any]       
    texto_clinico: Optional[str] = None 