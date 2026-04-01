"""
Инъекторы семантических аномалий.

Каждый класс реализует модификацию текстового содержимого спанов AEF Tracing
для имитации конкретного типа атаки/уязвимости по классификации LumiMAS.
"""

from __future__ import annotations

import json
import random
from typing import Optional

import pandas as pd

from .config import (
    AnomalyType, DPISubtype, InjectorConfig, GranularityMode,
)
from .base import BaseAnomalyInjector
from . import templates as T


class HallucinationInjector(BaseAnomalyInjector):
    """
    Инъекция галлюцинаций: модификация aef_output LLM-спанов
    добавлением вымышленных фактов или противоречий.
    """
    anomaly_type = AnomalyType.HALLUCINATION

    def inject_trace(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.hallucination
        df = trace_df.copy()

        mask = self._get_target_mask(df, cfg.target_kinds)
        if cfg.granularity == GranularityMode.ALL_SPANS:
            mask = pd.Series(True, index=df.index)

        if not mask.any():
            # Если нет подходящих спанов, модифицируем все что есть
            mask = pd.Series(True, index=df.index)

        for idx in df[mask].index:
            original = self._safe_get_text(df.iloc[df.index.get_loc(idx)], self.config.output_col)
            if not original:
                continue

            fabrication = T.pick_random(T.HALLUCINATION_FABRICATIONS, self.rng)

            # Иногда добавляем противоречие
            if self.rng.random() < 0.3:
                fabrication += ' ' + T.pick_random(T.HALLUCINATION_CONTRADICTIONS, self.rng)

            template = T.pick_random(T.HALLUCINATION_OUTPUT_TEMPLATES, self.rng)
            short_original = T.truncate_text(original, 500)
            new_output = T.render_template(template, original=short_original, fabrication=fabrication)

            df.at[idx, self.config.output_col] = new_output

            # Также модифицируем response_body если он есть
            resp_body = self._safe_get_text(df.iloc[df.index.get_loc(idx)], self.config.response_body_col)
            if resp_body:
                df.at[idx, self.config.response_body_col] = _inject_into_json_content(
                    resp_body, fabrication, self.rng
                )

        if cfg.perturb_epi:
            df = self._perturb_tokens(df, (0.8, 1.5), mask)

        return df


class BiasInjector(BaseAnomalyInjector):
    """
    Инъекция предвзятости: добавление стереотипных/дискриминационных
    суждений в выходные данные LLM-спанов.
    """
    anomaly_type = AnomalyType.BIAS

    def inject_trace(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.bias
        df = trace_df.copy()

        mask = self._get_target_mask(df, cfg.target_kinds)
        if cfg.granularity == GranularityMode.ALL_SPANS:
            mask = pd.Series(True, index=df.index)
        if not mask.any():
            mask = pd.Series(True, index=df.index)

        for idx in df[mask].index:
            original = self._safe_get_text(df.iloc[df.index.get_loc(idx)], self.config.output_col)
            if not original:
                continue

            bias_text = T.pick_random(T.BIAS_INJECTIONS, self.rng)
            template = T.pick_random(T.BIAS_OUTPUT_TEMPLATES, self.rng)
            short_original = T.truncate_text(original, 500)
            new_output = T.render_template(template, original=short_original, bias=bias_text)

            df.at[idx, self.config.output_col] = new_output

            resp_body = self._safe_get_text(df.iloc[df.index.get_loc(idx)], self.config.response_body_col)
            if resp_body:
                df.at[idx, self.config.response_body_col] = _inject_into_json_content(
                    resp_body, bias_text, self.rng
                )

        return df


class DPIInjector(BaseAnomalyInjector):
    """
    Direct Prompt Injection: модификация aef_input / aef_request_body
    LLM-спанов для имитации 3 подтипов атак.
    """
    anomaly_type = AnomalyType.DPI

    def __init__(self, config: InjectorConfig, rng: random.Random,
                 subtype: Optional[DPISubtype] = None):
        super().__init__(config, rng)
        self._forced_subtype = subtype

    @property
    def current_subtype(self) -> DPISubtype:
        if self._forced_subtype:
            return self._forced_subtype
        return self.rng.choice(list(self.config.dpi.subtypes))

    def inject_trace(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        subtype = self.current_subtype

        if subtype == DPISubtype.MISINFORMATION:
            return self._inject_misinformation(trace_df)
        elif subtype == DPISubtype.EXHAUSTION:
            return self._inject_exhaustion(trace_df)
        elif subtype == DPISubtype.BACKDOOR:
            return self._inject_backdoor(trace_df)
        else:
            raise ValueError(f'Неизвестный подтип DPI: {subtype}')

    def _inject_misinformation(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        """DPI-Misinformation: вставка ложных инструкций в промпт."""
        cfg = self.config.dpi
        df = trace_df.copy()
        mask = self._get_target_mask(df, cfg.target_kinds)
        if not mask.any():
            mask = pd.Series(True, index=df.index)

        for idx in df[mask].index:
            payload = T.pick_random(T.DPI_MISINFO_PAYLOADS, self.rng)

            # Модифицируем input
            original_input = self._safe_get_text(df.iloc[df.index.get_loc(idx)], self.config.input_col)
            template = T.pick_random(T.DPI_INPUT_TEMPLATES, self.rng)
            df.at[idx, self.config.input_col] = T.render_template(
                template, payload=payload, original=original_input
            )

            # Модифицируем request_body
            req_body = self._safe_get_text(df.iloc[df.index.get_loc(idx)], self.config.request_body_col)
            if req_body:
                df.at[idx, self.config.request_body_col] = _inject_into_json_content(
                    req_body, payload, self.rng
                )

        # Метка подтипа сохраняется в вызывающем коде
        df['_dpi_subtype'] = DPISubtype.MISINFORMATION.value
        return df

    def _inject_exhaustion(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        """DPI-Exhaustion: раздувание промпта для исчерпания ресурсов."""
        cfg = self.config.dpi
        df = trace_df.copy()
        mask = self._get_target_mask(df, cfg.target_kinds)
        if not mask.any():
            mask = pd.Series(True, index=df.index)

        for idx in df[mask].index:
            original_input = self._safe_get_text(df.iloc[df.index.get_loc(idx)], self.config.input_col)
            repeats = self.rng.randint(*cfg.exhaustion_prompt_repeat)
            filler = (T.DPI_EXHAUSTION_FILLER + '\n') * repeats
            df.at[idx, self.config.input_col] = filler + original_input

            req_body = self._safe_get_text(df.iloc[df.index.get_loc(idx)], self.config.request_body_col)
            if req_body:
                df.at[idx, self.config.request_body_col] = filler + req_body

        # EPI-возмущения: раздувание токенов и длительности
        df = self._perturb_tokens(df, cfg.exhaustion_token_factor, mask)
        df = self._perturb_duration(df, cfg.exhaustion_token_factor, mask)

        df['_dpi_subtype'] = DPISubtype.EXHAUSTION.value
        return df

    def _inject_backdoor(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        """DPI-Backdoor: вставка скрытых команд/триггеров."""
        cfg = self.config.dpi
        df = trace_df.copy()
        mask = self._get_target_mask(df, cfg.target_kinds)
        if not mask.any():
            mask = pd.Series(True, index=df.index)

        for idx in df[mask].index:
            payload = T.pick_random(T.DPI_BACKDOOR_PAYLOADS, self.rng)
            original_input = self._safe_get_text(df.iloc[df.index.get_loc(idx)], self.config.input_col)

            template = T.pick_random(T.DPI_INPUT_TEMPLATES, self.rng)
            df.at[idx, self.config.input_col] = T.render_template(
                template, payload=payload, original=original_input
            )

        df['_dpi_subtype'] = DPISubtype.BACKDOOR.value
        return df


class IPIInjector(BaseAnomalyInjector):
    """
    Indirect Prompt Injection: модификация выходных данных
    инструментов/внешних запросов для имитации атаки через
    внешние источники данных.
    """
    anomaly_type = AnomalyType.IPI

    def inject_trace(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.ipi
        df = trace_df.copy()

        mask = self._get_target_mask(df, cfg.target_kinds)

        # Если нет tool/retriever спанов, модифицируем chain как fallback
        if not mask.any():
            mask = self._get_target_mask(df, ('chain', 'llm'))
        if not mask.any():
            mask = pd.Series(True, index=df.index)

        for idx in df[mask].index:
            payload = T.pick_random(T.IPI_TOOL_OUTPUT_PAYLOADS, self.rng)
            original = self._safe_get_text(df.iloc[df.index.get_loc(idx)], self.config.output_col)
            template = T.pick_random(T.IPI_OUTPUT_TEMPLATES, self.rng)
            new_output = T.render_template(template, payload=payload, original=original)
            df.at[idx, self.config.output_col] = new_output

            # Также модифицируем response_body если есть
            resp_body = self._safe_get_text(df.iloc[df.index.get_loc(idx)], self.config.response_body_col)
            if resp_body:
                df.at[idx, self.config.response_body_col] = new_output

        return df


class MPInjector(BaseAnomalyInjector):
    """
    Memory Poisoning: модификация содержимого retriever/chain спанов
    для имитации отравления RAG-базы данных.
    """
    anomaly_type = AnomalyType.MP

    def inject_trace(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.mp
        df = trace_df.copy()

        mask = self._get_target_mask(df, cfg.target_kinds)

        # Если нет retriever спанов, модифицируем chain/llm как fallback
        if not mask.any():
            mask = self._get_target_mask(df, ('chain', 'llm'))
        if not mask.any():
            mask = pd.Series(True, index=df.index)

        for idx in df[mask].index:
            poisoned = T.pick_random(T.MP_POISONED_CONTENT, self.rng)
            original = self._safe_get_text(df.iloc[df.index.get_loc(idx)], self.config.output_col)
            template = T.pick_random(T.MP_OUTPUT_TEMPLATES, self.rng)
            new_output = T.render_template(template, poisoned=poisoned, original=original)
            df.at[idx, self.config.output_col] = new_output

            # Для retriever-спанов также модифицируем input (запрос в RAG)
            kind = self._safe_get_text(df.iloc[df.index.get_loc(idx)], self.config.span_kind_col)
            if kind == 'retriever':
                resp_body = self._safe_get_text(df.iloc[df.index.get_loc(idx)], self.config.response_body_col)
                if resp_body:
                    df.at[idx, self.config.response_body_col] = new_output

        return df


# ═══════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ═══════════════════════════════════════════════════════════════════════

def _inject_into_json_content(json_str: str, injection: str, rng: random.Random) -> str:
    """
    Попытаться распарсить JSON-строку и вставить текст инъекции
    в поле 'content' (если есть). Если парсинг невозможен — дописать в конец.
    """
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            if 'content' in data:
                data['content'] = str(data['content']) + ' ' + injection
            elif 'message' in data and isinstance(data['message'], dict):
                if 'content' in data['message']:
                    data['message']['content'] = str(data['message']['content']) + ' ' + injection
            return json.dumps(data, ensure_ascii=False)
        elif isinstance(data, list):
            # Модифицируем первый элемент с content
            for item in data:
                if isinstance(item, dict):
                    if 'content' in item:
                        item['content'] = str(item['content']) + ' ' + injection
                        break
                    elif 'message' in item and isinstance(item['message'], dict):
                        if 'content' in item['message']:
                            item['message']['content'] = str(item['message']['content']) + ' ' + injection
                            break
            return json.dumps(data, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    return json_str + ' ' + injection
