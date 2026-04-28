#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
DATE_TAG = "20260425"
OUT_ROOT = IMPORTANT / "optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle"
CONTRACTS = OUT_ROOT / "contracts"
TABLES = OUT_ROOT / "tables"
REPORTS = OUT_ROOT / "reports"
RUNS = OUT_ROOT / "runs"
RUNS_DIRECT_FITTING = RUNS / "direct_fitting"
RUNS_ACTIVE_SEQUENTIAL = RUNS / "active_sequential"

DEFAULT_TRACE_LENGTH = 2000
DEFAULT_DT_MS = 0.025
DEFAULT_PULSE_START_FRAC = 0.10
DEFAULT_PULSE_END_FRAC = 0.90
DEFAULT_G_SCALE_RANGE = (0.2, 5.0)
DEFAULT_TAU_SCALE_RANGE = (0.5, 2.0)
DEFAULT_GL_SCALE_RANGE = (0.1, 5.0)
DEFAULT_CURRENT_GAIN_BOUNDS = (2.0, 80.0)
DEFAULT_DOWNSAMPLE_POINTS = 128
OBSERVED_CURRENT_LABELS = ("0.1", "0.2", "0.3", "0.4", "0.5")
ACTIVE_CURRENT_GRID = tuple(f"{x:.2f}" for x in np.arange(0.05, 0.65, 0.05))

TRACK4_FEATURES = (
    REPO
    / "data"
    / "important_notes"
    / "optimization_track_20260411_v100_interactive"
    / "shared_data"
    / "track4_hh_full"
    / "concatenated_data"
    / "y"
)

RAW_PARAM_LOW = np.asarray([0.05, 0.2, 5.0, 0.05, 0.2, 5.0], dtype=np.float64)
RAW_PARAM_HIGH = np.asarray([10000.0, 100.0, 1000.0, 10000.0, 100.0, 1000.0], dtype=np.float64)


@dataclass(frozen=True)
class ParameterSchemaRow:
    parameter_index: int
    dataset_name: str
    assumption_level: str
    assumed_group: str
    assumed_role: str
    surrogate_mapping_target: str
    raw_low: float
    raw_high: float
    raw_scale: str
    mapped_interval_low: float
    mapped_interval_high: float
    literature_rationale: str


@dataclass(frozen=True)
class ProtocolContract:
    contract_name: str
    assumption_level: str
    waveform_family: str
    observed_current_labels: tuple[str, ...]
    candidate_active_current_grid: tuple[str, ...]
    current_label_interpretation: str
    trace_length: int
    dt_ms: float
    pulse_start_frac: float
    pulse_end_frac: float
    literature_rationale: str


@dataclass(frozen=True)
class ObservationContract:
    contract_name: str
    assumption_level: str
    source_path_pattern: str
    source_row_count: int
    source_feature_length: int
    surrogate_trace_length: int
    feature_representation: str
    downsample_points: int
    summary_representation: str
    literature_rationale: str


def ensure_bundle_dirs() -> None:
    for path in [CONTRACTS, TABLES, REPORTS, RUNS, RUNS_DIRECT_FITTING, RUNS_ACTIVE_SEQUENTIAL]:
        path.mkdir(parents=True, exist_ok=True)


def parameter_schema_rows() -> list[ParameterSchemaRow]:
    return [
        ParameterSchemaRow(
            parameter_index=1,
            dataset_name="hh_param_1",
            assumption_level="assumption_conditioned_not_provenance_recovered",
            assumed_group="block_1",
            assumed_role="fast_inward_conductance_scale_proxy",
            surrogate_mapping_target="g_na",
            raw_low=float(RAW_PARAM_LOW[0]),
            raw_high=float(RAW_PARAM_HIGH[0]),
            raw_scale="log10_uniform_support",
            mapped_interval_low=float(DEFAULT_G_SCALE_RANGE[0]),
            mapped_interval_high=float(DEFAULT_G_SCALE_RANGE[1]),
            literature_rationale="Compact HH reductions commonly vary inward conductance magnitude, and the workspace prior already allocates a repeated conductance-scale slot.",
        ),
        ParameterSchemaRow(
            parameter_index=2,
            dataset_name="hh_param_2",
            assumption_level="assumption_conditioned_not_provenance_recovered",
            assumed_group="block_1",
            assumed_role="fast_activation_kinetics_scale_proxy",
            surrogate_mapping_target="tau_m_scale",
            raw_low=float(RAW_PARAM_LOW[1]),
            raw_high=float(RAW_PARAM_HIGH[1]),
            raw_scale="log10_uniform_support",
            mapped_interval_low=float(DEFAULT_TAU_SCALE_RANGE[0]),
            mapped_interval_high=float(DEFAULT_TAU_SCALE_RANGE[1]),
            literature_rationale="Reduced HH inference often varies gating timescales, and the repeated low/high support pattern is consistent with a kinetics scale slot.",
        ),
        ParameterSchemaRow(
            parameter_index=3,
            dataset_name="hh_param_3",
            assumption_level="assumption_conditioned_not_provenance_recovered",
            assumed_group="block_1",
            assumed_role="slow_inactivation_kinetics_scale_proxy",
            surrogate_mapping_target="tau_h_scale",
            raw_low=float(RAW_PARAM_LOW[2]),
            raw_high=float(RAW_PARAM_HIGH[2]),
            raw_scale="log10_uniform_support",
            mapped_interval_low=float(DEFAULT_TAU_SCALE_RANGE[0]),
            mapped_interval_high=float(DEFAULT_TAU_SCALE_RANGE[1]),
            literature_rationale="Compact HH-style surrogates often separate fast and slow gating scales, and the third slot is treated as the slower block-1 kinetics term.",
        ),
        ParameterSchemaRow(
            parameter_index=4,
            dataset_name="hh_param_4",
            assumption_level="assumption_conditioned_not_provenance_recovered",
            assumed_group="block_2",
            assumed_role="outward_conductance_scale_proxy",
            surrogate_mapping_target="g_k",
            raw_low=float(RAW_PARAM_LOW[3]),
            raw_high=float(RAW_PARAM_HIGH[3]),
            raw_scale="log10_uniform_support",
            mapped_interval_low=float(DEFAULT_G_SCALE_RANGE[0]),
            mapped_interval_high=float(DEFAULT_G_SCALE_RANGE[1]),
            literature_rationale="A second conductance-scale block is consistent with compact HH inward-versus-outward channel separation and with the repeated support triplet.",
        ),
        ParameterSchemaRow(
            parameter_index=5,
            dataset_name="hh_param_5",
            assumption_level="assumption_conditioned_not_provenance_recovered",
            assumed_group="block_2",
            assumed_role="outward_activation_kinetics_scale_proxy",
            surrogate_mapping_target="tau_n_scale",
            raw_low=float(RAW_PARAM_LOW[4]),
            raw_high=float(RAW_PARAM_HIGH[4]),
            raw_scale="log10_uniform_support",
            mapped_interval_low=float(DEFAULT_TAU_SCALE_RANGE[0]),
            mapped_interval_high=float(DEFAULT_TAU_SCALE_RANGE[1]),
            literature_rationale="The second kinetics-scale slot is mapped to outward activation speed, which is the compact-HH role that matches the existing surrogate codepath.",
        ),
        ParameterSchemaRow(
            parameter_index=6,
            dataset_name="hh_param_6",
            assumption_level="assumption_conditioned_not_provenance_recovered",
            assumed_group="leak",
            assumed_role="leak_conductance_scale_proxy",
            surrogate_mapping_target="g_l",
            raw_low=float(RAW_PARAM_LOW[5]),
            raw_high=float(RAW_PARAM_HIGH[5]),
            raw_scale="log10_uniform_support",
            mapped_interval_low=float(DEFAULT_GL_SCALE_RANGE[0]),
            mapped_interval_high=float(DEFAULT_GL_SCALE_RANGE[1]),
            literature_rationale="A leak-scale slot is a standard compact HH degree of freedom, and the existing sandbox already maps the sixth slot to leak conductance.",
        ),
        ParameterSchemaRow(
            parameter_index=7,
            dataset_name="surrogate_current_gain",
            assumption_level="assumption_conditioned_not_provenance_recovered",
            assumed_group="protocol_gain",
            assumed_role="global_current_amplitude_gain_proxy",
            surrogate_mapping_target="current_gain",
            raw_low=float(DEFAULT_CURRENT_GAIN_BOUNDS[0]),
            raw_high=float(DEFAULT_CURRENT_GAIN_BOUNDS[1]),
            raw_scale="log10_uniform_support",
            mapped_interval_low=float(DEFAULT_CURRENT_GAIN_BOUNDS[0]),
            mapped_interval_high=float(DEFAULT_CURRENT_GAIN_BOUNDS[1]),
            literature_rationale="Because the physical units of the observed current labels are unknown, a global gain term is introduced so the surrogate protocol can vary on a relative scale.",
        ),
    ]


def protocol_contract() -> ProtocolContract:
    return ProtocolContract(
        contract_name="hh_track4_surrogate_current_clamp_protocol",
        assumption_level="assumption_conditioned_not_provenance_recovered",
        waveform_family="single square-pulse current clamp",
        observed_current_labels=OBSERVED_CURRENT_LABELS,
        candidate_active_current_grid=ACTIVE_CURRENT_GRID,
        current_label_interpretation="Observed labels and candidate amplitudes are treated as relative current multipliers, rather than recovered physical units.",
        trace_length=DEFAULT_TRACE_LENGTH,
        dt_ms=DEFAULT_DT_MS,
        pulse_start_frac=DEFAULT_PULSE_START_FRAC,
        pulse_end_frac=DEFAULT_PULSE_END_FRAC,
        literature_rationale="Square-pulse current clamp is a standard neuron-model fitting protocol, and step-current sweeps are common in both BluePyOpt-style optimization and experimental-design studies.",
    )


def observation_contract() -> ObservationContract:
    return ObservationContract(
        contract_name="hh_track4_surrogate_voltage_observation",
        assumption_level="assumption_conditioned_not_provenance_recovered",
        source_path_pattern=str(TRACK4_FEATURES / "concatenated_data_{curr}_curr.npy"),
        source_row_count=15000,
        source_feature_length=400000,
        surrogate_trace_length=DEFAULT_TRACE_LENGTH,
        feature_representation="standardized_downsample_plus_summary12",
        downsample_points=DEFAULT_DOWNSAMPLE_POINTS,
        summary_representation="summary12",
        literature_rationale="Compact HH inverse problems are normally driven by voltage traces, while summary features remain useful for optimization and discrepancy scoring when provenance is incomplete.",
    )


def claim_guardrails() -> dict[str, Any]:
    return {
        "label_required": "assumption_conditioned_not_provenance_recovered",
        "allowed_uses": [
            "exploratory surrogate direct fitting",
            "assumption-conditioned active sequential design",
            "workflow prototyping for later provenance recovery",
            "sensitivity analysis over protocol and support assumptions",
        ],
        "disallowed_claims": [
            "recovered Track4 simulator provenance",
            "dataset-faithful direct fitting",
            "true physical current units for labels 0.1 to 0.5",
            "true semantic identity of hh_param_1 to hh_param_6",
        ],
        "related_existing_scripts": [
            str(REPO / "tools" / "run_hh_track4_assumption_conditioned_hybrid_refinement_20260425.py"),
            str(REPO / "tools" / "run_hh_track4_assumption_conditioned_wasserstein_abc_20260425.py"),
        ],
    }


def combined_contract_manifest() -> dict[str, Any]:
    return {
        "date_tag": DATE_TAG,
        "bundle_root": str(OUT_ROOT),
        "assumption_conditioned": True,
        "provenance_recovered": False,
        "parameter_schema": [asdict(row) for row in parameter_schema_rows()],
        "protocol_contract": asdict(protocol_contract()),
        "observation_contract": asdict(observation_contract()),
        "claim_guardrails": claim_guardrails(),
    }


def json_ready(data: Any) -> Any:
    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, dict):
        return {str(k): json_ready(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [json_ready(v) for v in data]
    return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n")

