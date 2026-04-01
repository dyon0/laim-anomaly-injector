"""
Конфигурация модуля инъекции аномалий для MAS-monitor.

Определяет параметры инъекции для каждого типа аномалий по классификации LumiMAS:
- EPI (Execution Performance Indicators): числовые/структурные возмущения
- Semantic: текстовые возмущения (Hallucination, Bias, DPI, IPI, MP)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional


class AnomalyType(str, Enum):
    """Типы аномалий по классификации LumiMAS."""
    HALLUCINATION = 'Hallucination'
    BIAS          = 'Bias'
    DPI           = 'DPI'
    IPI           = 'IPI'
    MP            = 'MP'


class DPISubtype(str, Enum):
    """Подтипы DPI-атак (LumiMAS Section 3 / Appendix C.3)."""
    MISINFORMATION = 'DPI-Misinformation'
    EXHAUSTION     = 'DPI-Exhaustion'
    BACKDOOR       = 'DPI-Backdoor'


class GranularityMode(str, Enum):
    """Режим гранулярности инъекции внутри трейса."""
    RELEVANT_SPANS_ONLY = 'relevant'   # модифицируем только релевантные спаны
    ALL_SPANS           = 'all'        # модифицируем все спаны в трейсе


# ─── Конфигурации для каждого типа аномалии ────────────────────────────

@dataclass(frozen=True)
class EPIPerturbationConfig:
    """Параметры EPI-возмущений (числовые/структурные)."""
    # Множители для длительности спанов
    duration_factor_range: tuple[float, float] = (2.0, 10.0)
    # Множители для токенов
    token_factor_range: tuple[float, float] = (2.0, 8.0)
    # Вероятность добавления дополнительных LLM-спанов
    extra_span_prob: float = 0.3
    # Количество дополнительных спанов (если добавляем)
    extra_spans_range: tuple[int, int] = (1, 3)
    # Вероятность удаления спана
    drop_span_prob: float = 0.15


@dataclass(frozen=True)
class HallucinationConfig:
    """Параметры инъекции галлюцинаций."""
    granularity: GranularityMode = GranularityMode.RELEVANT_SPANS_ONLY
    # Какие aef_kind затрагиваем
    target_kinds: tuple[str, ...] = ('llm', 'chain')
    # Модифицировать ли также EPI-признаки (токены и т.д.)
    perturb_epi: bool = False


@dataclass(frozen=True)
class BiasConfig:
    """Параметры инъекции предвзятости."""
    granularity: GranularityMode = GranularityMode.RELEVANT_SPANS_ONLY
    target_kinds: tuple[str, ...] = ('llm', 'chain')
    perturb_epi: bool = False


@dataclass(frozen=True)
class DPIConfig:
    """Параметры DPI-атак (все 3 подтипа)."""
    granularity: GranularityMode = GranularityMode.RELEVANT_SPANS_ONLY
    target_kinds: tuple[str, ...] = ('llm',)
    subtypes: tuple[DPISubtype, ...] = (
        DPISubtype.MISINFORMATION,
        DPISubtype.EXHAUSTION,
        DPISubtype.BACKDOOR,
    )
    # Для Exhaustion: множитель раздувания промпта
    exhaustion_prompt_repeat: tuple[int, int] = (5, 20)
    # Для Exhaustion: множитель токенов
    exhaustion_token_factor: tuple[float, float] = (5.0, 15.0)
    perturb_epi: bool = True  # DPI-Exhaustion требует изменения EPI


@dataclass(frozen=True)
class IPIConfig:
    """Параметры IPI-атак (через внешние источники)."""
    granularity: GranularityMode = GranularityMode.RELEVANT_SPANS_ONLY
    target_kinds: tuple[str, ...] = ('tool', 'retriever', 'output_request')
    perturb_epi: bool = False


@dataclass(frozen=True)
class MPConfig:
    """Параметры Memory Poisoning атак."""
    granularity: GranularityMode = GranularityMode.RELEVANT_SPANS_ONLY
    target_kinds: tuple[str, ...] = ('retriever', 'chain', 'tool')
    perturb_epi: bool = False


# ─── Главная конфигурация ────────────────────────────────────────────

@dataclass
class InjectorConfig:
    """
    Главная конфигурация инъектора аномалий.

    Attributes:
        anomaly_ratio: доля аномальных трейсов в итоговом датасете (0.0 – 1.0)
        anomaly_types: какие типы аномалий генерировать
        anomaly_weights: веса типов (для распределения); если None — равномерно
        random_seed: сид для воспроизводимости
        clone_traces: клонировать трейсы для аномалий (True) или модифицировать оригиналы (False)
    """
    anomaly_ratio: float = 0.5
    anomaly_types: tuple[AnomalyType, ...] = (
        AnomalyType.HALLUCINATION,
        AnomalyType.BIAS,
        AnomalyType.DPI,
        AnomalyType.IPI,
        AnomalyType.MP,
    )
    anomaly_weights: Optional[tuple[float, ...]] = None
    random_seed: int = 42
    clone_traces: bool = True

    # Конфигурации по типам
    epi: EPIPerturbationConfig = field(default_factory=EPIPerturbationConfig)
    hallucination: HallucinationConfig = field(default_factory=HallucinationConfig)
    bias: BiasConfig = field(default_factory=BiasConfig)
    dpi: DPIConfig = field(default_factory=DPIConfig)
    ipi: IPIConfig = field(default_factory=IPIConfig)
    mp: MPConfig = field(default_factory=MPConfig)

    # Колонки AEF CSV
    trace_id_col: str = 'traceid'
    span_kind_col: str = 'aef_kind'
    start_nano_col: str = 'starttimeunixnano'
    end_nano_col: str = 'endtimeunixnano'
    output_col: str = 'aef_output'
    input_col: str = 'aef_input'
    request_body_col: str = 'aef_request_body'
    response_body_col: str = 'aef_response_body'
    prompt_tokens_col: str = 'aef_llm_prompt_tokens'
    completion_tokens_col: str = 'aef_llm_completion_tokens'
    total_tokens_col: str = 'aef_llm_total_tokens'
    precached_tokens_col: str = 'aef_llm_precached_prompt_tokens'
    model_params_col: str = 'aef_llm_model_parameters'
    metadata_col: str = 'aef_metadata'
    span_id_col: str = 'spanid'
    name_col: str = 'name'
    status_col: str = 'status'

    # Колонки-метки (будут добавлены)
    label_col: str = 'label'
    anomaly_type_col: str = 'anomaly_type'
    dpi_subtype_col: str = 'dpi_subtype'
