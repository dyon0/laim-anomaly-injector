"""Базовый класс инъектора аномалий."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from .config import AnomalyType, InjectorConfig


class BaseAnomalyInjector(ABC):
    """
    Абстрактный базовый класс для инъекторов конкретных типов аномалий.

    Каждый подкласс реализует inject_trace(), который принимает DataFrame
    одного трейса (все спаны с одним traceid) и возвращает модифицированную копию.
    """

    anomaly_type: AnomalyType

    def __init__(self, config: InjectorConfig, rng: random.Random):
        self.config = config
        self.rng = rng

    @abstractmethod
    def inject_trace(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        """
        Инъецировать аномалию в трейс.

        Args:
            trace_df: DataFrame со спанами одного трейса (копия оригинала).

        Returns:
            Модифицированный DataFrame с инъецированной аномалией.
        """
        ...

    def _get_target_mask(self, trace_df: pd.DataFrame, target_kinds: tuple[str, ...]) -> pd.Series:
        """Получить маску спанов, подлежащих модификации по aef_kind."""
        kind_col = self.config.span_kind_col
        return trace_df[kind_col].isin(target_kinds)

    def _safe_get_text(self, row: pd.Series, col: str) -> str:
        """Безопасно извлечь текстовое значение из строки."""
        val = row.get(col)
        if pd.isna(val) or val is None:
            return ''
        return str(val)

    def _perturb_duration(self, trace_df: pd.DataFrame, factor_range: tuple[float, float],
                          mask: Optional[pd.Series] = None) -> pd.DataFrame:
        """Умножить длительность спанов на случайный множитель."""
        start_col = self.config.start_nano_col
        end_col = self.config.end_nano_col

        if mask is None:
            mask = pd.Series(True, index=trace_df.index)

        df = trace_df.copy()
        for idx in df[mask].index:
            factor = self.rng.uniform(*factor_range)
            start = pd.to_numeric(df.at[idx, start_col], errors='coerce')
            end = pd.to_numeric(df.at[idx, end_col], errors='coerce')
            if pd.notna(start) and pd.notna(end):
                duration = end - start
                new_duration = int(duration * factor)
                df.at[idx, end_col] = int(start + new_duration)
        return df

    def _perturb_tokens(self, trace_df: pd.DataFrame, factor_range: tuple[float, float],
                        mask: Optional[pd.Series] = None) -> pd.DataFrame:
        """Умножить счётчики токенов на случайный множитель."""
        token_cols = [
            self.config.prompt_tokens_col,
            self.config.completion_tokens_col,
            self.config.total_tokens_col,
        ]

        if mask is None:
            mask = pd.Series(True, index=trace_df.index)

        df = trace_df.copy()
        for idx in df[mask].index:
            factor = self.rng.uniform(*factor_range)
            for col in token_cols:
                val = pd.to_numeric(df.at[idx, col], errors='coerce')
                if pd.notna(val) and val > 0:
                    df.at[idx, col] = int(val * factor)
        return df
