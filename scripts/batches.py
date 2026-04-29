from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class BatchStep:
    step_id: str
    kind: str
    description: str
    params_file: str
    mode: Optional[str] = None
    baseline_name: Optional[str] = None
    feature_mode: Optional[str] = None
    sbi_method: Optional[str] = None
    density_estimator: Optional[str] = None
    pass_curr: bool = False
    save_predictions: Optional[str] = None
    load_from_step: Optional[str] = None
    split_eval_after_train: bool = False
    skip_inline_split_eval: bool = False
    n_train: Optional[int] = None
    eval_limit: Optional[int] = None
    posterior_samples: Optional[int] = None
    stop_after_epochs: Optional[int] = None
    max_num_epochs: Optional[int] = None
    seed: Optional[int] = None
    nproc_per_node: Optional[int] = None
    step_nnodes: Optional[int] = None
    extra_args: Tuple[str, ...] = ()
    env_overrides: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class BatchDefinition:
    batch_id: str
    sequence: int
    title: str
    suite: str
    scientific_role: str
    summary: str
    requires_hh_tar: bool = False
    requires_slurm: bool = False
    cluster_label: Optional[str] = None
    min_nodes: int = 0
    preferred_gpu: Optional[str] = None
    depends_on: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()
    steps: Tuple[BatchStep, ...] = ()


BATCHES: Tuple[BatchDefinition, ...] = (
    BatchDefinition(
        batch_id="smk01_fhn_baseline",
        sequence=10,
        title="FitzHugh-Nagumo baseline smoke",
        suite="falcon_a30_smoke",
        scientific_role="supporting",
        summary="Runs the shipped FHN smoke train and eval path, so the public code surface can be validated before any HH workload is attempted.",
        requires_slurm=True,
        cluster_label="Falcon",
        min_nodes=4,
        preferred_gpu="A30",
        steps=(
            BatchStep(
                step_id="train",
                kind="dnn",
                description="FHN smoke training run",
                params_file="src/pytorch/configs/params_dnn_tar_smoke.yaml",
                mode="train",
                save_predictions="none",
            ),
            BatchStep(
                step_id="eval",
                kind="dnn",
                description="FHN smoke evaluation run against the latest checkpoint from the preceding train step",
                params_file="src/pytorch/configs/params_dnn_tar_smoke.yaml",
                mode="eval",
                load_from_step="train",
                save_predictions="none",
            ),
        ),
    ),
    BatchDefinition(
        batch_id="smk02_hh_dnn",
        sequence=20,
        title="Hodgkin-Huxley DNN smoke",
        suite="falcon_a30_smoke",
        scientific_role="primary_support",
        summary="Exercises the HH DNN train and eval path from a fresh checkout, using only the external HH tarball and the automatic `.prepared_data` extraction flow.",
        requires_hh_tar=True,
        requires_slurm=True,
        cluster_label="Falcon",
        min_nodes=4,
        preferred_gpu="A30",
        notes=(
            "This batch validates the public HH DNN path, rather than the full canonical campaign scale.",
        ),
        steps=(
            BatchStep(
                step_id="train",
                kind="dnn",
                description="HH smoke training run",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                mode="train",
                pass_curr=True,
                save_predictions="none",
            ),
            BatchStep(
                step_id="eval",
                kind="dnn",
                description="HH smoke evaluation run against the latest checkpoint from the preceding train step",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                mode="eval",
                pass_curr=True,
                load_from_step="train",
                save_predictions="none",
            ),
        ),
    ),
    BatchDefinition(
        batch_id="smk03_hh_classical",
        sequence=30,
        title="Hodgkin-Huxley classical smoke",
        suite="falcon_a30_smoke",
        scientific_role="primary_support",
        summary="Runs the public classical HH baseline driver on the smoke contract, so the non-neural public path can be validated independently of DNN training.",
        requires_hh_tar=True,
        requires_slurm=True,
        cluster_label="Falcon",
        min_nodes=4,
        preferred_gpu="A30",
        steps=(
            BatchStep(
                step_id="extra_trees_200_raw",
                kind="classical",
                description="Extra Trees smoke reference on raw trace features",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                baseline_name="extra_trees_200",
                feature_mode="raw",
                pass_curr=True,
            ),
        ),
    ),
    BatchDefinition(
        batch_id="smk04_hh_sbi",
        sequence=40,
        title="Hodgkin-Huxley SBI smoke",
        suite="falcon_a30_smoke",
        scientific_role="boundary_support",
        summary="Runs the public HH SBI driver on a compact SNPE smoke contract, so the posterior-oriented public path can be validated independently of the main DNN and classical paths.",
        requires_hh_tar=True,
        requires_slurm=True,
        cluster_label="Falcon",
        min_nodes=4,
        preferred_gpu="A30",
        steps=(
            BatchStep(
                step_id="snpe_maf_raw",
                kind="sbi",
                description="SNPE-MAF smoke reference on raw trace features",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                sbi_method="snpe",
                density_estimator="maf",
                feature_mode="raw",
                pass_curr=True,
                n_train=64,
                eval_limit=64,
                posterior_samples=8,
                stop_after_epochs=1,
                max_num_epochs=1,
                seed=123,
            ),
        ),
    ),
    BatchDefinition(
        batch_id="fal01_hh_4node_a30_dnn",
        sequence=50,
        title="Falcon 4-node A30 HH DNN canonical run",
        suite="canonical_falcon",
        scientific_role="primary",
        summary="Runs the current public distributed HH DNN entrypoint inside a Falcon Slurm allocation, using the canonical 4-node A30 placeholder topology and a separate train and eval step.",
        requires_hh_tar=True,
        requires_slurm=True,
        cluster_label="Falcon",
        min_nodes=4,
        preferred_gpu="A30",
        notes=(
            "This batch is environment-anchored. It is intended for cluster reproduction, rather than standalone workstation execution.",
        ),
        steps=(
            BatchStep(
                step_id="train",
                kind="interactive_dnn",
                description="Distributed HH DNN train step inside a 4-node Falcon allocation",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh.yaml",
                pass_curr=True,
                save_predictions="none",
                split_eval_after_train=True,
                skip_inline_split_eval=True,
                step_nnodes=4,
                nproc_per_node=2,
            ),
            BatchStep(
                step_id="eval",
                kind="interactive_dnn",
                description="Post-train evaluation step against the latest checkpoint from the preceding distributed HH DNN train step",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh.yaml",
                pass_curr=True,
                save_predictions="none",
                load_from_step="train",
                step_nnodes=4,
                nproc_per_node=2,
            ),
        ),
    ),
    BatchDefinition(
        batch_id="fal02_hh_classical_reference",
        sequence=60,
        title="Falcon HH classical reference",
        suite="canonical_falcon",
        scientific_role="primary_support",
        summary="Runs the public classical HH baseline driver under the documented Falcon environment, so the classical public reference can be regenerated within the same cluster contract.",
        requires_hh_tar=True,
        requires_slurm=True,
        cluster_label="Falcon",
        min_nodes=1,
        preferred_gpu="A30",
        steps=(
            BatchStep(
                step_id="extra_trees_200_raw",
                kind="classical",
                description="Cluster-side Extra Trees reference on raw trace features",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                baseline_name="extra_trees_200",
                feature_mode="raw",
                pass_curr=True,
            ),
        ),
    ),
    BatchDefinition(
        batch_id="fal03_hh_sbi_reference",
        sequence=70,
        title="Falcon HH SBI reference",
        suite="canonical_falcon",
        scientific_role="boundary_support",
        summary="Runs the public HH SBI driver inside the documented Falcon environment, so the posterior-oriented public reference can be regenerated within the same cluster contract.",
        requires_hh_tar=True,
        requires_slurm=True,
        cluster_label="Falcon",
        min_nodes=1,
        preferred_gpu="A30",
        steps=(
            BatchStep(
                step_id="snpe_maf_raw",
                kind="sbi",
                description="Cluster-side SNPE-MAF reference on raw trace features",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                sbi_method="snpe",
                density_estimator="maf",
                feature_mode="raw",
                pass_curr=True,
                n_train=64,
                eval_limit=64,
                posterior_samples=8,
                stop_after_epochs=1,
                max_num_epochs=1,
                seed=123,
            ),
        ),
    ),
    BatchDefinition(
        batch_id="sci01_benchmark_parity",
        sequence=110,
        title="Scientific smoke: benchmark parity",
        suite="scientific_smoke",
        scientific_role="supporting",
        summary="Reduced but scientifically meaningful FHN smoke that validates the baseline train and eval path used to anchor the benchmark-parity thread.",
        requires_slurm=True,
        cluster_label="Falcon",
        min_nodes=4,
        preferred_gpu="A30",
        steps=(
            BatchStep(
                step_id="train",
                kind="dnn",
                description="FHN parity-smoke training run",
                params_file="src/pytorch/configs/params_dnn_tar_smoke.yaml",
                mode="train",
                save_predictions="none",
            ),
            BatchStep(
                step_id="eval",
                kind="dnn",
                description="FHN parity-smoke evaluation run",
                params_file="src/pytorch/configs/params_dnn_tar_smoke.yaml",
                mode="eval",
                load_from_step="train",
                save_predictions="none",
            ),
        ),
    ),
    BatchDefinition(
        batch_id="sci02_execution_policy",
        sequence=120,
        title="Scientific smoke: execution configuration and run policy",
        suite="scientific_smoke",
        scientific_role="supporting",
        summary="Reduced HH distributed smoke across 1-node, 2-node, and 4-node Falcon layouts, so the execution-policy thread can be exercised through the same distributed public entrypoint.",
        requires_hh_tar=True,
        requires_slurm=True,
        cluster_label="Falcon",
        min_nodes=4,
        preferred_gpu="A30",
        depends_on=("sci01_benchmark_parity",),
        steps=(
            BatchStep(
                step_id="one_node_train",
                kind="interactive_dnn",
                description="Single-node HH distributed smoke train",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                pass_curr=True,
                save_predictions="none",
                split_eval_after_train=True,
                skip_inline_split_eval=True,
                step_nnodes=1,
                nproc_per_node=1,
                env_overrides=(("DATA_ACCESS_MODE", "tar_in_place"),),
            ),
            BatchStep(
                step_id="one_node_eval",
                kind="interactive_dnn",
                description="Single-node HH distributed smoke eval",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                pass_curr=True,
                save_predictions="none",
                load_from_step="one_node_train",
                step_nnodes=1,
                nproc_per_node=1,
                env_overrides=(("DATA_ACCESS_MODE", "tar_in_place"),),
            ),
            BatchStep(
                step_id="two_node_train",
                kind="interactive_dnn",
                description="Two-node HH distributed smoke train",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                pass_curr=True,
                save_predictions="none",
                split_eval_after_train=True,
                skip_inline_split_eval=True,
                step_nnodes=2,
                nproc_per_node=1,
                env_overrides=(("DATA_ACCESS_MODE", "tar_in_place"),),
            ),
            BatchStep(
                step_id="two_node_eval",
                kind="interactive_dnn",
                description="Two-node HH distributed smoke eval",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                pass_curr=True,
                save_predictions="none",
                load_from_step="two_node_train",
                step_nnodes=2,
                nproc_per_node=1,
                env_overrides=(("DATA_ACCESS_MODE", "tar_in_place"),),
            ),
            BatchStep(
                step_id="four_node_train",
                kind="interactive_dnn",
                description="Four-node HH distributed smoke train",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                pass_curr=True,
                save_predictions="none",
                split_eval_after_train=True,
                skip_inline_split_eval=True,
                step_nnodes=4,
                nproc_per_node=2,
                env_overrides=(("DATA_ACCESS_MODE", "tar_in_place"),),
            ),
            BatchStep(
                step_id="four_node_eval",
                kind="interactive_dnn",
                description="Four-node HH distributed smoke eval",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                pass_curr=True,
                save_predictions="none",
                load_from_step="four_node_train",
                step_nnodes=4,
                nproc_per_node=2,
                env_overrides=(("DATA_ACCESS_MODE", "tar_in_place"),),
            ),
        ),
    ),
    BatchDefinition(
        batch_id="sci03_direct_frontier",
        sequence=130,
        title="Scientific smoke: direct fixed-task frontier",
        suite="scientific_smoke",
        scientific_role="primary",
        summary="Reduced HH direct-comparison smoke across the public neural and classical representatives, so the direct frontier can be regenerated at smoke scale.",
        requires_hh_tar=True,
        requires_slurm=True,
        cluster_label="Falcon",
        min_nodes=4,
        preferred_gpu="A30",
        depends_on=("sci02_execution_policy",),
        steps=(
            BatchStep(
                step_id="dnn_train",
                kind="dnn",
                description="Neural direct smoke train",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                mode="train",
                pass_curr=True,
                save_predictions="none",
            ),
            BatchStep(
                step_id="dnn_eval",
                kind="dnn",
                description="Neural direct smoke eval",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                mode="eval",
                pass_curr=True,
                load_from_step="dnn_train",
                save_predictions="none",
            ),
            BatchStep(
                step_id="extra_trees_clean",
                kind="classical",
                description="Extra Trees direct clean smoke reference",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                baseline_name="extra_trees_500_depth20",
                feature_mode="raw",
                pass_curr=True,
            ),
            BatchStep(
                step_id="knn_clean",
                kind="classical",
                description="kNN direct clean smoke reference",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                baseline_name="knn_k11",
                feature_mode="raw",
                pass_curr=True,
            ),
        ),
    ),
    BatchDefinition(
        batch_id="sci04_posterior_followup",
        sequence=140,
        title="Scientific smoke: posterior and representation follow-up",
        suite="scientific_smoke",
        scientific_role="boundary",
        summary="Reduced HH posterior smoke across representation and family choices, so the representation-enrichment and posterior-family thread can be exercised publicly.",
        requires_hh_tar=True,
        requires_slurm=True,
        cluster_label="Falcon",
        min_nodes=4,
        preferred_gpu="A30",
        depends_on=("sci03_direct_frontier",),
        steps=(
            BatchStep(
                step_id="snpe_raw",
                kind="sbi",
                description="SNPE-MAF raw-trace smoke reference",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                sbi_method="snpe",
                density_estimator="maf",
                feature_mode="raw",
                pass_curr=True,
                n_train=64,
                eval_limit=64,
                posterior_samples=8,
                stop_after_epochs=1,
                max_num_epochs=1,
                seed=123,
            ),
            BatchStep(
                step_id="snpe_raw_fft_summary",
                kind="sbi",
                description="SNPE-MAF smoke with enriched raw-plus-FFT-plus-summary features",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                sbi_method="snpe",
                density_estimator="maf",
                feature_mode="raw_plus_fft256_summary12",
                pass_curr=True,
                n_train=64,
                eval_limit=64,
                posterior_samples=8,
                stop_after_epochs=1,
                max_num_epochs=1,
                seed=123,
            ),
            BatchStep(
                step_id="fmpe_raw_fft_summary",
                kind="sbi",
                description="FMPE smoke with enriched raw-plus-FFT-plus-summary features",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                sbi_method="fmpe",
                density_estimator="maf",
                feature_mode="raw_plus_fft256_summary12",
                pass_curr=True,
                n_train=64,
                eval_limit=64,
                posterior_samples=8,
                stop_after_epochs=1,
                max_num_epochs=1,
                seed=123,
            ),
            BatchStep(
                step_id="snpe_median_rule",
                kind="sbi",
                description="SNPE-MAF smoke with a posterior-median decision rule",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                sbi_method="snpe",
                density_estimator="maf",
                feature_mode="raw_plus_fft256_summary12",
                pass_curr=True,
                n_train=64,
                eval_limit=64,
                posterior_samples=8,
                stop_after_epochs=1,
                max_num_epochs=1,
                seed=123,
                extra_args=("--decision-rule", "median"),
            ),
        ),
    ),
    BatchDefinition(
        batch_id="sci05_robustness",
        sequence=150,
        title="Scientific smoke: robustness and perturbation response",
        suite="scientific_smoke",
        scientific_role="boundary",
        summary="Reduced HH robustness smoke that exercises the public direct and posterior pathways under additive noise, drift, and masking.",
        requires_hh_tar=True,
        requires_slurm=True,
        cluster_label="Falcon",
        min_nodes=4,
        preferred_gpu="A30",
        depends_on=("sci04_posterior_followup",),
        steps=(
            BatchStep(
                step_id="extra_trees_noise020",
                kind="classical",
                description="Extra Trees smoke under additive noise 0.20",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                baseline_name="extra_trees_500_depth20",
                feature_mode="raw",
                pass_curr=True,
                extra_args=(
                    "--features-additive-noise-std",
                    "0.20",
                    "--random-seed",
                    "123",
                ),
            ),
            BatchStep(
                step_id="knn_noise020",
                kind="classical",
                description="kNN smoke under additive noise 0.20",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                baseline_name="knn_k11",
                feature_mode="raw",
                pass_curr=True,
                extra_args=(
                    "--features-additive-noise-std",
                    "0.20",
                    "--random-seed",
                    "123",
                ),
            ),
            BatchStep(
                step_id="snpe_drift010",
                kind="sbi",
                description="SNPE-MAF smoke with matched drift retraining and drift evaluation",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                sbi_method="snpe",
                density_estimator="maf",
                feature_mode="raw_plus_fft256_summary12",
                pass_curr=True,
                n_train=64,
                eval_limit=64,
                posterior_samples=8,
                stop_after_epochs=1,
                max_num_epochs=1,
                seed=123,
                extra_args=(
                    "--train-baseline-drift-std",
                    "0.10",
                    "--eval-baseline-drift-std",
                    "0.10",
                ),
            ),
            BatchStep(
                step_id="snpe_mask20",
                kind="sbi",
                description="SNPE-MAF smoke with matched masking retraining and masking evaluation",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                sbi_method="snpe",
                density_estimator="maf",
                feature_mode="raw_plus_fft256_summary12",
                pass_curr=True,
                n_train=64,
                eval_limit=64,
                posterior_samples=8,
                stop_after_epochs=1,
                max_num_epochs=1,
                seed=123,
                extra_args=(
                    "--train-mask-fraction",
                    "0.20",
                    "--eval-mask-fraction",
                    "0.20",
                ),
            ),
        ),
    ),
    BatchDefinition(
        batch_id="sci06_public_extensions",
        sequence=160,
        title="Scientific smoke: public extension coverage",
        suite="scientific_smoke",
        scientific_role="coverage_expansion",
        summary="Reduced public extension smoke that exercises the later method and decision-rule surface shipped in the public repo, which serves as the repository-side proxy for the broader extension thread.",
        requires_hh_tar=True,
        requires_slurm=True,
        cluster_label="Falcon",
        min_nodes=4,
        preferred_gpu="A30",
        depends_on=("sci05_robustness",),
        notes=(
            "The public repository does not ship the private later-framework code paths such as BayesFlow, Swyft, or active-sequential design. This batch therefore covers the later public posterior-family and decision-rule surface that is actually shipped.",
        ),
        steps=(
            BatchStep(
                step_id="fmpe_extension",
                kind="sbi",
                description="FMPE extension smoke on enriched features",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                sbi_method="fmpe",
                density_estimator="maf",
                feature_mode="raw_plus_fft256_summary12",
                pass_curr=True,
                n_train=64,
                eval_limit=64,
                posterior_samples=8,
                stop_after_epochs=1,
                max_num_epochs=1,
                seed=123,
            ),
            BatchStep(
                step_id="npse_extension",
                kind="sbi",
                description="NPSE extension smoke on enriched features",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                sbi_method="npse",
                density_estimator="maf",
                feature_mode="raw_plus_fft256_summary12",
                pass_curr=True,
                n_train=64,
                eval_limit=64,
                posterior_samples=8,
                stop_after_epochs=1,
                max_num_epochs=1,
                seed=123,
                extra_args=("--device", "cpu"),
            ),
            BatchStep(
                step_id="snpe_validate_calibrated",
                kind="sbi",
                description="SNPE smoke with validation-calibrated decision-rule selection",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml",
                sbi_method="snpe",
                density_estimator="maf",
                feature_mode="raw_plus_fft256_summary12",
                pass_curr=True,
                n_train=64,
                eval_limit=64,
                posterior_samples=8,
                stop_after_epochs=1,
                max_num_epochs=1,
                seed=123,
                extra_args=(
                    "--decision-rule",
                    "validate_calibrated",
                    "--selection-candidate-rules",
                    "mean,median",
                ),
            ),
        ),
    ),
    BatchDefinition(
        batch_id="sci07_execution_path",
        sequence=170,
        title="Scientific smoke: execution-path optimizations",
        suite="scientific_smoke",
        scientific_role="engineering_support",
        summary="Reduced HH cache-path smoke on Falcon, so the public execution-path optimizations can be exercised through the distributed launcher with cache-enabled public settings.",
        requires_hh_tar=True,
        requires_slurm=True,
        cluster_label="Falcon",
        min_nodes=4,
        preferred_gpu="A30",
        depends_on=("sci02_execution_policy",),
        steps=(
            BatchStep(
                step_id="cache_copy_to_node_train",
                kind="interactive_dnn",
                description="Cache-enabled HH smoke train with node-local tar extraction",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke_cache.yaml",
                pass_curr=True,
                save_predictions="none",
                split_eval_after_train=True,
                skip_inline_split_eval=True,
                step_nnodes=1,
                nproc_per_node=1,
                env_overrides=(("DATA_ACCESS_MODE", "copy_to_node"),),
            ),
            BatchStep(
                step_id="cache_copy_to_node_eval",
                kind="interactive_dnn",
                description="Cache-enabled HH smoke eval with node-local tar extraction",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke_cache.yaml",
                pass_curr=True,
                save_predictions="none",
                load_from_step="cache_copy_to_node_train",
                step_nnodes=1,
                nproc_per_node=1,
                env_overrides=(("DATA_ACCESS_MODE", "copy_to_node"),),
            ),
            BatchStep(
                step_id="cache_tar_in_place_train",
                kind="interactive_dnn",
                description="Cache-enabled HH smoke train with tar-in-place access",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke_cache.yaml",
                pass_curr=True,
                save_predictions="none",
                split_eval_after_train=True,
                skip_inline_split_eval=True,
                step_nnodes=1,
                nproc_per_node=1,
                env_overrides=(("DATA_ACCESS_MODE", "tar_in_place"),),
            ),
            BatchStep(
                step_id="cache_tar_in_place_eval",
                kind="interactive_dnn",
                description="Cache-enabled HH smoke eval with tar-in-place access",
                params_file="src/pytorch/configs/hh/params_dnn_tar_hh_smoke_cache.yaml",
                pass_curr=True,
                save_predictions="none",
                load_from_step="cache_tar_in_place_train",
                step_nnodes=1,
                nproc_per_node=1,
                env_overrides=(("DATA_ACCESS_MODE", "tar_in_place"),),
            ),
        ),
    ),
)


SUITES: Dict[str, Tuple[str, ...]] = {
    "falcon_a30_smoke": (
        "smk01_fhn_baseline",
        "smk02_hh_dnn",
        "smk03_hh_classical",
        "smk04_hh_sbi",
    ),
    "canonical_falcon": (
        "fal01_hh_4node_a30_dnn",
        "fal02_hh_classical_reference",
        "fal03_hh_sbi_reference",
    ),
    "scientific_smoke": (
        "sci01_benchmark_parity",
        "sci02_execution_policy",
        "sci03_direct_frontier",
        "sci04_posterior_followup",
        "sci05_robustness",
        "sci06_public_extensions",
        "sci07_execution_path",
    ),
}


BATCH_BY_ID: Dict[str, BatchDefinition] = {batch.batch_id: batch for batch in BATCHES}
ORDERED_BATCH_IDS: Tuple[str, ...] = tuple(batch.batch_id for batch in sorted(BATCHES, key=lambda item: item.sequence))


def get_batch(batch_id: str) -> BatchDefinition:
    return BATCH_BY_ID[batch_id]


def list_batches() -> Tuple[BatchDefinition, ...]:
    return tuple(sorted(BATCHES, key=lambda item: item.sequence))


def batches_for_suite(suite: str) -> Tuple[BatchDefinition, ...]:
    batch_ids = SUITES[suite]
    return tuple(get_batch(batch_id) for batch_id in batch_ids)


def batches_through(batch_id: str) -> Tuple[BatchDefinition, ...]:
    target = get_batch(batch_id)
    return tuple(batch for batch in list_batches() if batch.sequence <= target.sequence)


def batch_to_dict(batch: BatchDefinition) -> Dict[str, object]:
    out = asdict(batch)
    out["steps"] = [asdict(step) for step in batch.steps]
    return out


def suite_names() -> Tuple[str, ...]:
    return tuple(sorted(SUITES.keys()))


def iter_dependency_chain(batch: BatchDefinition) -> Iterable[str]:
    for dep in batch.depends_on:
        yield dep
