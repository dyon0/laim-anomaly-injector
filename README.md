# anomaly_injector — Модуль инъекции аномалий для MAS-monitor

## Назначение

Механизм синтетической инъекции аномалий в трейсы AI-агентов (формат AEF Tracing CSV) для калибровки порогов детекции аномалий в детекторе аномалий.

## Типы аномалий

Реализованы все 5 типов уязвимостей по классификации LumiMAS:

| Тип | Метка | Что модифицируется | Целевые спаны |
|-----|-------|--------------------|---------------|
| **Hallucination** | `Hallucination` | `aef_output`, `aef_response_body` — вставка вымышленных фактов/противоречий | `llm`, `chain` |
| **Bias** | `Bias` | `aef_output`, `aef_response_body` — добавление стереотипных суждений | `llm`, `chain` |
| **DPI-Misinformation** | `DPI` | `aef_input`, `aef_request_body` — ложные инструкции в промпт | `llm` |
| **DPI-Exhaustion** | `DPI` | `aef_input` + EPI (токены ×5-15, длительность ×5-15) | `llm` |
| **DPI-Backdoor** | `DPI` | `aef_input` — скрытые команды/триггеры | `llm` |
| **IPI** | `IPI` | `aef_output` инструментов — малициозный контент из «внешних источников» | `tool`, `retriever`, `output_request` |
| **MP** | `MP` | `aef_output` — отравленные данные из RAG/памяти | `retriever`, `chain`, `tool` |

## Архитектура

```
anomaly_injector/
├── __init__.py          # Публичный API
├── config.py            # Конфигурации (dataclasses)
├── templates.py         # Русскоязычные шаблоны текстовых инъекций
├── injector.py          # Оркестратор AnomalyInjector
└── injectors/
    ├── __init__.py
    ├── base.py          # Базовый класс BaseAnomalyInjector
    ├── semantic.py      # Hallucination, Bias, DPI, IPI, MP injectors
    └── epi.py           # EPIPerturbator (числовые/структурные возмущения)
```

## Быстрый старт

```python
from anomaly_injector import AnomalyInjector, InjectorConfig

# Конфигурация
config = InjectorConfig(
    anomaly_ratio=0.5,        # 50% трейсов будут аномальными
    clone_traces=True,        # клонировать (оригиналы сохранятся как нормальные)
    random_seed=42,
)

# Инъекция
injector = AnomalyInjector(config)
result = injector.inject_and_save(
    'traces.csv',                    # входной файл (CSV или Parquet)
    'traces_with_anomalies.csv',     # выходной файл
)

# Или функциональный API
from anomaly_injector import inject_anomalies
import pandas as pd

df = pd.read_csv('traces.csv')
result = inject_anomalies(df, anomaly_ratio=0.5)
```

## Выходной формат

К исходным колонкам AEF CSV добавляются:

| Колонка | Тип | Описание |
|---------|-----|----------|
| `label` | bool | `True` = аномалия, `False` = норма |
| `anomaly_type` | str \| None | `Hallucination`, `Bias`, `DPI`, `IPI`, `MP` |
| `dpi_subtype` | str \| None | `DPI-Misinformation`, `DPI-Exhaustion`, `DPI-Backdoor` |

## Конфигурация по типам

```python
config = InjectorConfig(
    # Глобальные параметры
    anomaly_ratio=0.5,
    anomaly_types=(AnomalyType.DPI, AnomalyType.IPI),  # только DPI и IPI
    anomaly_weights=(0.7, 0.3),                          # 70% DPI, 30% IPI

    # EPI-возмущения
    epi=EPIPerturbationConfig(
        duration_factor_range=(2.0, 10.0),
        token_factor_range=(2.0, 8.0),
    ),

    # DPI с конкретными подтипами
    dpi=DPIConfig(
        subtypes=(DPISubtype.EXHAUSTION, DPISubtype.MISINFORMATION),
        exhaustion_token_factor=(5.0, 15.0),
        granularity=GranularityMode.RELEVANT_SPANS_ONLY,
    ),
)
```

## Интеграция с MAS-monitor pipeline

```python
# В build_dataset.py или data.py:
from anomaly_injector import inject_anomalies

# 1. Загрузить сырые трейсы
df = pd.read_csv('raw_traces.csv')

# 2. Инъецировать аномалии
df_with_anomalies = inject_anomalies(df, anomaly_ratio=0.5)

# 3. Передать в адаптер AEF -> MAS-monitor spans
from adapter.aef_adapter import TraceAdapter
adapter = TraceAdapter(df_with_anomalies, MAS_COLUMN_MAPPING)
spans = adapter.transform()

# 4. Метки label/anomaly_type сохраняются через pipeline
```
