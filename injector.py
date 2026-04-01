"""
Главный оркестратор инъекции аномалий.

AnomalyInjector принимает датасет трейсов AEF в формате CSV/DataFrame,
распределяет трейсы по типам аномалий и применяет соответствующие инъекторы.

Основной цикл:
1. Группировка спанов по traceid
2. Разделение трейсов на «нормальные» и «целевые для инъекции»
3. Клонирование целевых трейсов (если clone_traces=True)
4. Применение инъекторов к каждому целевому трейсу
5. Конкатенация результатов с метками label / anomaly_type
6. Экспорт в CSV / Parquet
"""

from __future__ import annotations

import logging
import random
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .config import (
    AnomalyType, DPISubtype, InjectorConfig,
)
from .base import BaseAnomalyInjector
from .semantic import (
    HallucinationInjector,
    BiasInjector,
    DPIInjector,
    IPIInjector,
    MPInjector,
)
from .epi import EPIPerturbator


logger = logging.getLogger(__name__)


class AnomalyInjector:
    """
    Оркестратор инъекции аномалий в трейсы AEF Tracing.

    Usage:
        config = InjectorConfig(anomaly_ratio=0.5)
        injector = AnomalyInjector(config)

        # Из файла
        result = injector.inject_from_file('traces.csv')
        result.to_csv('traces_with_anomalies.csv')

        # Из DataFrame
        df = pd.read_csv('traces.csv')
        result = injector.inject(df)
    """

    def __init__(self, config: Optional[InjectorConfig] = None):
        self.config = config or InjectorConfig()
        self.rng = random.Random(self.config.random_seed)
        self._epi = EPIPerturbator(self.config, self.rng)
        self._injectors = self._build_injectors()

    def _build_injectors(self) -> Dict[AnomalyType, BaseAnomalyInjector]:
        """Создать инъекторы для каждого сконфигурированного типа аномалий."""
        mapping: Dict[AnomalyType, type] = {
            AnomalyType.HALLUCINATION: HallucinationInjector,
            AnomalyType.BIAS:          BiasInjector,
            AnomalyType.DPI:           DPIInjector,
            AnomalyType.IPI:           IPIInjector,
            AnomalyType.MP:            MPInjector,
        }
        injectors = {}
        for atype in self.config.anomaly_types:
            cls = mapping.get(atype)
            if cls is None:
                raise ValueError(f'Неизвестный тип аномалии: {atype}')
            injectors[atype] = cls(self.config, self.rng)
        return injectors

    def inject(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Инъецировать аномалии в датасет трейсов.

        Args:
            df: DataFrame с AEF-спанами (формат strategy_builder_*.csv)

        Returns:
            DataFrame с добавленными колонками:
            - label: bool (True = аномалия)
            - anomaly_type: str | None (тип аномалии по LumiMAS)
            - dpi_subtype: str | None (подтип DPI, если применимо)
        """
        trace_col = self.config.trace_id_col
        label_col = self.config.label_col
        atype_col = self.config.anomaly_type_col
        dpi_col = self.config.dpi_subtype_col

        # Группируем по traceId
        trace_ids = df[trace_col].unique().tolist()
        n_traces = len(trace_ids)
        n_anomalous = max(1, int(n_traces * self.config.anomaly_ratio))

        logger.info(
            f'Всего трейсов: {n_traces}, '
            f'планируемых аномальных: {n_anomalous} '
            f'(ratio={self.config.anomaly_ratio:.2f})'
        )

        # Распределяем типы аномалий
        assignments = self._assign_anomaly_types(trace_ids, n_anomalous)

        normal_dfs: List[pd.DataFrame] = []
        anomalous_dfs: List[pd.DataFrame] = []

        for tid in trace_ids:
            trace_mask = df[trace_col] == tid
            trace_df = df[trace_mask].copy()

            if tid not in assignments:
                # Нормальный трейс
                trace_df[label_col] = False
                trace_df[atype_col] = None
                trace_df[dpi_col] = None
                normal_dfs.append(trace_df)
            else:
                atype = assignments[tid]

                if self.config.clone_traces:
                    # Сохраняем оригинал как нормальный
                    orig = trace_df.copy()
                    orig[label_col] = False
                    orig[atype_col] = None
                    orig[dpi_col] = None
                    normal_dfs.append(orig)

                    # Создаём клон с новым traceid
                    cloned = trace_df.copy()
                    new_tid = f'{tid}_anom_{uuid.uuid4().hex[:8]}'
                    cloned[trace_col] = new_tid
                else:
                    cloned = trace_df

                # Применяем инъектор
                injector = self._injectors[atype]
                injected = injector.inject_trace(cloned)

                # Если у типа аномалии есть EPI-возмущения и инъектор их не сделал сам
                if atype == AnomalyType.DPI and self.config.dpi.perturb_epi:
                    pass  # DPI-Exhaustion уже применяет EPI в injector
                # Можно добавить доп. EPI для других типов если нужно

                # Проставляем метки
                injected[label_col] = True
                injected[atype_col] = atype.value

                # DPI подтип
                if '_dpi_subtype' in injected.columns:
                    injected[dpi_col] = injected['_dpi_subtype']
                    injected = injected.drop(columns=['_dpi_subtype'])
                else:
                    injected[dpi_col] = None

                anomalous_dfs.append(injected)

        # Собираем результат
        all_dfs = normal_dfs + anomalous_dfs
        result = pd.concat(all_dfs, ignore_index=True)

        # Статистика
        n_normal = result[result[label_col] == False].groupby(trace_col).ngroups
        n_anom = result[result[label_col] == True].groupby(trace_col).ngroups
        logger.info(f'Результат: {n_normal} нормальных + {n_anom} аномальных трейсов')

        type_counts = (
            result[result[label_col] == True]
            .groupby(atype_col)[trace_col]
            .nunique()
            .to_dict()
        )
        for at, cnt in type_counts.items():
            logger.info(f'  {at}: {cnt} трейсов')

        return result

    def _assign_anomaly_types(
        self,
        trace_ids: List[str],
        n_anomalous: int,
    ) -> Dict[str, AnomalyType]:
        """
        Распределить трейсы по типам аномалий.

        Returns:
            Словарь {trace_id -> AnomalyType} для аномальных трейсов.
        """
        types = list(self.config.anomaly_types)
        weights = self.config.anomaly_weights

        if weights is None:
            weights = tuple(1.0 / len(types) for _ in types)

        # Выбираем трейсы для инъекции
        selected = self.rng.sample(trace_ids, min(n_anomalous, len(trace_ids)))

        # Распределяем типы с учётом весов
        assignments = {}
        for tid in selected:
            atype = self.rng.choices(types, weights=weights, k=1)[0]
            assignments[tid] = atype

        return assignments

    # ─── Файловый API ──────────────────────────────────────────────

    def inject_from_file(self, input_path: str | Path) -> pd.DataFrame:
        """Загрузить трейсы из CSV/Parquet, инъецировать аномалии."""
        path = Path(input_path)
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
        else:
            raise ValueError(f'Неподдерживаемый формат файла: {path.suffix}')

        return self.inject(df)

    def inject_and_save(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> pd.DataFrame:
        """Загрузить, инъецировать и сохранить в указанный файл."""
        result = self.inject_from_file(input_path)
        output = Path(output_path)

        if output.suffix == '.parquet':
            result.to_parquet(output, index=False)
        elif output.suffix == '.csv':
            result.to_csv(output, index=False)
        else:
            # Сохраняем оба формата
            result.to_csv(output.with_suffix('.csv'), index=False)
            result.to_parquet(output.with_suffix('.parquet'), index=False)

        logger.info(f'Результат сохранён: {output}')
        return result


def inject_anomalies(
    df: pd.DataFrame,
    config: Optional[InjectorConfig] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Функциональный API: инъецировать аномалии в датасет.

    Args:
        df: DataFrame с AEF-спанами
        config: конфигурация (если None — используются значения по умолчанию)
        **kwargs: дополнительные параметры для InjectorConfig

    Returns:
        DataFrame с инъецированными аномалиями и метками.

    Example:
        >>> from anomaly_injector import inject_anomalies
        >>> df = pd.read_csv('traces.csv')
        >>> result = inject_anomalies(df, anomaly_ratio=0.5)
    """
    if config is None:
        config = InjectorConfig(**kwargs) if kwargs else InjectorConfig()
    injector = AnomalyInjector(config)
    return injector.inject(df)
