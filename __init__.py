"""
anomaly_injector — модуль инъекции аномалий для MAS-monitor.

Механизм инъекции аномалий в трейсы AI-агентов в формате AEF Tracing.
Реализует все типы аномалий по классификации LumiMAS:
- Hallucination (галлюцинации LLM)
- Bias (предвзятость)
- DPI (Direct Prompt Injection): Misinformation, Exhaustion, Backdoor
- IPI (Indirect Prompt Injection)
- MP (Memory Poisoning)

Каждый тип аномалии может затрагивать как семантические (текстовые),
так и EPI (числовые/структурные) характеристики трейсов.

Usage:
    from anomaly_injector import AnomalyInjector, InjectorConfig

    config = InjectorConfig(anomaly_ratio=0.5)
    injector = AnomalyInjector(config)
    result = injector.inject_from_file('traces.csv')
    result.to_csv('traces_with_anomalies.csv')

    # Или функциональный API:
    from anomaly_injector import inject_anomalies
    result = inject_anomalies(df, anomaly_ratio=0.5)
"""

from .config import (
    AnomalyType,
    DPISubtype,
    GranularityMode,
    InjectorConfig,
    EPIPerturbationConfig,
    HallucinationConfig,
    BiasConfig,
    DPIConfig,
    IPIConfig,
    MPConfig,
)
from .injector import AnomalyInjector, inject_anomalies

__version__ = '0.1.0'

__all__ = [
    'AnomalyInjector',
    'inject_anomalies',
    'InjectorConfig',
    'AnomalyType',
    'DPISubtype',
    'GranularityMode',
    'EPIPerturbationConfig',
    'HallucinationConfig',
    'BiasConfig',
    'DPIConfig',
    'IPIConfig',
    'MPConfig',
]
