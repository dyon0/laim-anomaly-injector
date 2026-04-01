"""
Инъекторы EPI-аномалий (структурные и числовые возмущения).

EPI-аномалии не привязаны к конкретному типу атаки — они моделируют
отклонения в числовых/структурных характеристиках выполнения агента,
которые могут быть побочным эффектом любой атаки или сбоя.

Используются как дополнительный компонент для тех типов аномалий,
у которых perturb_epi=True (например, DPI-Exhaustion).
"""

from __future__ import annotations

import random
import uuid
from typing import Optional

import pandas as pd

from .config import InjectorConfig, EPIPerturbationConfig


class EPIPerturbator:
    """
    Утилитарный класс для применения EPI-возмущений к трейсу.

    Реализует три вида возмущений:
    1. Temporal: изменение длительности спанов
    2. Token: изменение счётчиков токенов
    3. Structural: добавление/удаление спанов
    """

    def __init__(self, config: InjectorConfig, rng: random.Random):
        self.config = config
        self.epi_cfg = config.epi
        self.rng = rng

    def perturb_duration(self, trace_df: pd.DataFrame,
                         factor_range: Optional[tuple[float, float]] = None,
                         mask: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Увеличить длительность спанов в factor раз.
        Имитирует замедление из-за exhaustion-атак или перегрузки.
        """
        if factor_range is None:
            factor_range = self.epi_cfg.duration_factor_range

        start_col = self.config.start_nano_col
        end_col = self.config.end_nano_col
        df = trace_df.copy()

        if mask is None:
            mask = pd.Series(True, index=df.index)

        for idx in df[mask].index:
            factor = self.rng.uniform(*factor_range)
            start = pd.to_numeric(df.at[idx, start_col], errors='coerce')
            end = pd.to_numeric(df.at[idx, end_col], errors='coerce')
            if pd.notna(start) and pd.notna(end) and end > start:
                duration = end - start
                new_end = int(start + duration * factor)
                df.at[idx, end_col] = new_end
        return df

    def perturb_tokens(self, trace_df: pd.DataFrame,
                       factor_range: Optional[tuple[float, float]] = None,
                       mask: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Умножить счётчики токенов на случайный множитель.
        Имитирует раздувание промптов или аномальную генерацию.
        """
        if factor_range is None:
            factor_range = self.epi_cfg.token_factor_range

        token_cols = [
            self.config.prompt_tokens_col,
            self.config.completion_tokens_col,
            self.config.total_tokens_col,
        ]

        df = trace_df.copy()
        if mask is None:
            mask = pd.Series(True, index=df.index)

        for idx in df[mask].index:
            factor = self.rng.uniform(*factor_range)
            for col in token_cols:
                val = pd.to_numeric(df.at[idx, col], errors='coerce')
                if pd.notna(val) and val > 0:
                    df.at[idx, col] = max(1, int(val * factor))
        return df

    def add_extra_spans(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавить дополнительные LLM-спаны в трейс.
        Имитирует лишние итерации агента (зацикливание, повторные вызовы).
        """
        if self.rng.random() > self.epi_cfg.extra_span_prob:
            return trace_df

        df = trace_df.copy()
        n_extra = self.rng.randint(*self.epi_cfg.extra_spans_range)

        # Берём за основу существующий LLM-спан (или любой спан)
        llm_mask = df[self.config.span_kind_col] == 'llm'
        if llm_mask.any():
            template_row = df[llm_mask].iloc[-1].copy()
        else:
            template_row = df.iloc[-1].copy()

        new_rows = []
        last_end = pd.to_numeric(template_row.get(self.config.end_nano_col, 0), errors='coerce')
        if pd.isna(last_end):
            last_end = 0

        for i in range(n_extra):
            new_row = template_row.copy()
            # Генерируем новые идентификаторы
            new_row[self.config.span_id_col] = uuid.uuid4().hex[:16]

            # Сдвигаем время
            gap = self.rng.randint(100_000, 10_000_000)  # 0.1ms - 10ms в нано
            new_start = int(last_end + gap)
            duration = self.rng.randint(50_000_000, 2_000_000_000)  # 50ms - 2s
            new_end = new_start + duration

            new_row[self.config.start_nano_col] = new_start
            new_row[self.config.end_nano_col] = new_end
            new_row[self.config.span_kind_col] = 'llm'
            new_row[self.config.name_col] = f'ExtraLLMCall_{i}'

            # Генерируем токены
            base_tokens = self.rng.randint(50, 500)
            new_row[self.config.prompt_tokens_col] = base_tokens
            new_row[self.config.completion_tokens_col] = self.rng.randint(20, 200)
            new_row[self.config.total_tokens_col] = base_tokens + self.rng.randint(20, 200)

            new_rows.append(new_row)
            last_end = new_end

        if new_rows:
            extra_df = pd.DataFrame(new_rows)
            df = pd.concat([df, extra_df], ignore_index=True)

        return df

    def drop_spans(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        """
        Удалить случайные спаны из трейса.
        Имитирует пропущенные шаги или прерванное выполнение.
        """
        if self.rng.random() > self.epi_cfg.drop_span_prob:
            return trace_df

        df = trace_df.copy()
        if len(df) <= 1:
            return df

        # Удаляем не более 30% спанов, но хотя бы 1
        n_drop = max(1, int(len(df) * 0.3 * self.rng.random()))
        n_drop = min(n_drop, len(df) - 1)  # оставляем хотя бы 1

        drop_indices = self.rng.sample(list(df.index), n_drop)
        df = df.drop(drop_indices).reset_index(drop=True)

        return df

    def apply_all(self, trace_df: pd.DataFrame,
                  mask: Optional[pd.Series] = None) -> pd.DataFrame:
        """Применить все EPI-возмущения последовательно."""
        df = self.perturb_duration(trace_df, mask=mask)
        df = self.perturb_tokens(df, mask=mask)
        df = self.add_extra_spans(df)
        df = self.drop_spans(df)
        return df
