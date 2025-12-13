#!/usr/bin/env python3
#===- act/pipeline/validate_verifier.py - Verifier Correctness Validation ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Unified verification validation framework with two validation levels:
#
#   Level 1: Counterexample/Soundness Validation
#     - Validates that verifier doesn't claim CERTIFIED when concrete 
#       counterexamples exist
#
#   Level 2: Bounds/Numerical Validation
#     - Validates that abstract bounds correctly overapproximate concrete 
#       activation values
#
#===---------------------------------------------------------------------===#
#
# Level 1: Counterexample/Soundness Validation
# ============================================
#
# Key Insight:
#   Concrete execution provides ground truth - if we find a real counterexample
#   at runtime, the formal verifier cannot claim the property is certified.
#   This is a soundness check for the verification backend.
#
# Validation Strategy:
#   1. For each network, generate strategic test cases:
#      - Center: Input at center of input spec (typically safe)
#      - Boundary: Input near boundary of input spec (risky)
#      - Random: Random input within input spec (varied)
#
#   2. Run concrete execution to find violations
#   3. If counterexample found, run formal verification
#   4. Cross-validate using matrix below
#
# Validation Matrix (Level 1):
#   ┌─────────────────────────┬────────────────────────────────────┬──────────────┐
#   │ Concrete Counterexample │ Verifier Result                    │ Validation   │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ FOUND                   │ CERTIFIED                          │ ❌ FAILED    │
#   │                         │ (Soundness Bug - false negative)   │              │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ FOUND                   │ FALSIFIED                          │ ✅ PASSED    │
#   │                         │ (Correct - verifier found issue)   │              │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ FOUND                   │ UNKNOWN                            │ ⚠️ ACCEPTABLE│
#   │                         │ (Incomplete but sound)             │              │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ NOT FOUND               │ Any Result                         │ ❓ INCONC.   │
#   │                         │ (Cannot validate - no ground truth)│              │
#   └─────────────────────────┴────────────────────────────────────┴──────────────┘
#
#   Legend:
#     FAILED       - Critical soundness bug (false negative)
#     PASSED       - Verifier correct
#     ACCEPTABLE   - Verifier incomplete but sound (conservative)
#     INCONCLUSIVE - No concrete counterexample to validate against
#
#===---------------------------------------------------------------------===#
#
# Level 2: Bounds/Numerical Validation
# ====================================
#
# Key Insight:
#   Abstract interpretation must overapproximate concrete values. If any
#   concrete activation value falls outside its abstract bounds [lb, ub],
#   the transfer function is unsound.
#
# Validation Strategy:
#   1. Sample concrete inputs from input specification
#   2. Run concrete forward pass through PyTorch model → get concrete activations
#   3. Run abstract analysis through ACT → get abstract bounds for each layer
#   4. Check: concrete_value ∈ [lb, ub] for all layers and all neurons
#
# Validation Matrix (Level 2):
#   ┌──────────────────────┬────────────────────────┬──────────────┐
#   │ Concrete Values      │ Abstract Bounds        │ Validation   │
#   ├──────────────────────┼────────────────────────┼──────────────┤
#   │ value ∈ [lb, ub]     │ All layers/neurons     │ ✅ PASSED    │
#   │ (Sound bounds)       │                        │              │
#   ├──────────────────────┼────────────────────────┼──────────────┤
#   │ value ∉ [lb, ub]     │ Any layer/neuron       │ ❌ FAILED    │
#   │ (Unsound bounds)     │ (Transfer function bug)│              │
#   └──────────────────────┴────────────────────────┴──────────────┘
#
#   Legend:
#     PASSED - All concrete values within abstract bounds (sound)
#     FAILED - Concrete value outside bounds (unsound transfer function)
#
#===---------------------------------------------------------------------===#
#
# Usage:
#   # Via CLI (recommended):
#   python -m act.pipeline --validate-verifier --mode comprehensive
#   python -m act.pipeline --validate-verifier --mode counterexample
#   python -m act.pipeline --validate-verifier --mode bounds
#   
#   # With device and dtype specification:
#   python -m act.pipeline --validate-verifier --device cpu --dtype float64
#   python -m act.pipeline --validate-verifier --device cuda --dtype float32
#   
#   # Test specific networks:
#   python -m act.pipeline --validate-verifier --networks mnist_mlp_small
#   python -m act.pipeline --validate-verifier --networks mnist_mlp_small,mnist_cnn_small
#   
#   # Test with specific solvers (Level 1):
#   python -m act.pipeline --validate-verifier --mode counterexample --solvers gurobi
#   python -m act.pipeline --validate-verifier --mode counterexample --solvers gurobi torchlp
#   
#   # Test with transfer function modes (Level 2):
#   python -m act.pipeline --validate-verifier --mode bounds --tf-modes interval
#   python -m act.pipeline --validate-verifier --mode bounds --tf-modes interval hybridz
#   
#   # Adjust number of samples for bounds validation:
#   python -m act.pipeline --validate-verifier --mode bounds --samples 20
#   
#   # Ignore errors and always exit 0 (useful for CI):
#   python -m act.pipeline --validate-verifier --ignore-errors
#   
#   # Combined options:
#   python -m act.pipeline --validate-verifier --mode comprehensive \
#       --networks mnist_mlp_small,mnist_cnn_small \
#       --solvers gurobi --tf-modes interval --samples 10 \
#       --device cpu --dtype float64
#
#   # Direct execution (legacy):
#   python act/pipeline/verification/validate_verifier.py
#   python act/pipeline/verification/validate_verifier.py --mode bounds --samples 5
#
# Exit Codes:
#   0 - All validations passed (no failures or errors)
#   0 - With --ignore-errors flag (always succeed regardless of results)
#   1 - Failures detected (verifier bugs) OR errors detected (backend bugs)
#
#===---------------------------------------------------------------------===#

from curses import meta
import os

# Mitigate OpenMP aborts seen on macOS when torch initializes multiple
# runtimes (Abort trap in libomp during import). Set conservative defaults
# before importing torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from act.pipeline.verification.model_factory import ModelFactory
from act.pipeline.verification.torch2act import TorchToACT
from act.back_end.verifier import verify_once, gather_input_spec_layers, seed_from_input_specs, get_input_ids, get_assert_layer, find_entry_layer_id
from act.back_end.analyze import analyze
from act.back_end.solver.solver_gurobi import GurobiSolver
from act.back_end.solver.solver_torch import TorchLPSolver
from act.util.options import PerformanceOptions
from act.front_end.specs import OutKind

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerificationValidator:
    """Unified verification validation framework with counterexample and bounds validation."""
    
    def __init__(
        self, 
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64
    ):
        """
        Initialize verification validator.
        
        Args:
            device: Device for computation ('cpu' or 'cuda')
            dtype: Data type for computation (float32 or float64)
        """
        self.factory = ModelFactory()
        self.device = device
        self.dtype = dtype
        self.validation_results = []
        
        # Initialize debug file (GUARDED)
        if PerformanceOptions.debug_tf:
            debug_file = PerformanceOptions.debug_output_file
            with open(debug_file, 'w') as f:
                f.write(f"ACT Verification Debug Log\n")
                f.write(f"Device: {device}, Dtype: {dtype}\n")
                f.write(f"{'='*80}\n\n")
            logger.info(f"Debug logging to: {debug_file}")
    
    # def find_concrete_counterexample(
    #     self, 
    #     name: str, 
    #     model: torch.nn.Module
    # ) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
    #     """
    #     Try to find a concrete counterexample through inference testing.
        
    #     Args:
    #         name: Network name
    #         model: PyTorch model for concrete execution
            
    #     Returns:
    #         (counterexample_input, results_dict) if found, None otherwise
    #     """
    #     test_cases = ['center', 'boundary', 'random']
        
    #     for test_case in test_cases:
    #         input_tensor = self.factory.generate_test_input(name, test_case)
    #         input_tensor = input_tensor.to(device=self.device, dtype=self.dtype)
    #         results = model(input_tensor)
            
    #         # Check if this is a counterexample
    #         if isinstance(results, dict):
    #             if results['input_satisfied'] and not results['output_satisfied']:
    #                 logger.info(f"  🔴 Counterexample found in '{test_case}' test case")
    #                 logger.info(f"     Input explanation: {results['input_explanation']}")
    #                 logger.info(f"     Output explanation: {results['output_explanation']}")
    #                 return (input_tensor, results)
        
    #     return None
    
    def find_concrete_counterexample(
        self,
        name: str,
        model: torch.nn.Module,
        act_net=None,
        max_random: int = 64,
    ) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Try to find a concrete counterexample through inference testing.

        现在版本：优先利用 ACT 的 INPUT_SPEC (lb/ub) 做系统搜索：
          1) 中心点
          2) 低维时的小网格 (3^d 点，d<=4)
          3) 中维时的 per-dimension 边界点 (d<=16)
          4) 一批随机点 (uniform in [lb, ub])
        找不到再退回旧的 center/boundary/random 逻辑。

        Args:
            name:   Network name
            model:  PyTorch model for concrete execution
            act_net: 对应的 ACT Net（可选，但建议传）
            max_random: 随机采样的数量上限

        Returns:
            (counterexample_input, results_dict) if found, None otherwise
        """
        # 小工具：统一做一次 forward + 判定
        def _check_point(x_flat: torch.Tensor, tag: str):
            """
            x_flat: 展平的输入向量（和 seed.lb 同 shape）
            返回 (tensor_in_shape, results) 或 None
            """
            # 推断输入 shape（来自 ACT 的 INPUT layer）
            input_shape = None
            if act_net is not None:
                for layer in act_net.layers:
                    if getattr(layer, "kind", None) == "INPUT":
                        input_shape = tuple(layer.meta.get("shape") or [])
                        break
            if input_shape is None:
                # 没有 meta 的话，退化成 [1, dim] 的 batch 输入
                input_shape = (1, x_flat.numel())

            try:
                x = x_flat.view(*input_shape)
            except Exception as e:
                logger.warning(
                    "  [CE search] reshape flat -> %s 失败 (%s)，退化成 batch=1, flat",
                    input_shape, e,
                )
                x = x_flat.view(1, -1)

            x = x.to(device=self.device, dtype=self.dtype)
            results = model(x)

            if isinstance(results, dict):
                in_sat = bool(results.get("input_satisfied", False))
                out_sat = bool(results.get("output_satisfied", True))
                if in_sat and (not out_sat):
                    logger.info(f"  🔴 Counterexample found ({tag})")
                    logger.info(f"     Input explanation:  {results.get('input_explanation')}")
                    logger.info(f"     Output explanation: {results.get('output_explanation')}")
                    return x, results
            return None

        # ============================================================
        # 1. 如果有 ACT net，就尝试从 INPUT_SPEC 提取 [lb, ub]
        # ============================================================
        try:
            if act_net is not None:
                # 复用 verifier 里已经引入的工具函数
                specs = gather_input_spec_layers(act_net)
                if specs:
                    seed = seed_from_input_specs(specs)
                    lb = seed.lb.to(device=self.device, dtype=self.dtype).flatten()
                    ub = seed.ub.to(device=self.device, dtype=self.dtype).flatten()
                    assert lb.shape == ub.shape
                    dim = lb.numel()

                    logger.info(
                        "  [CE search] Using INPUT_SPEC bounds for sampling: dim=%d, "
                        "lb[min,max]=[%.4f, %.4f], ub[min,max]=[%.4f, %.4f]",
                        dim,
                        float(lb.min().item()), float(lb.max().item()),
                        float(ub.min().item()), float(ub.max().item()),
                    )

                    # -------- 1) 中心点 --------
                    center = 0.5 * (lb + ub)
                    ce = _check_point(center, tag="center_from_spec")
                    if ce is not None:
                        return ce

                    # -------- 2) 低维网格：3^d（d<=4）--------
                    # 网格点 = {lb, (lb+ub)/2, ub}^d
                    if dim <= 4:
                        import itertools
                        grid_levels = torch.tensor([0.0, 0.5, 1.0], device=self.device, dtype=self.dtype)
                        for coeffs in itertools.product(range(3), repeat=dim):
                            # coeffs ∈ {0,1,2}^d
                            alphas = grid_levels[list(coeffs)]  # shape [dim]
                            x_flat = lb + alphas * (ub - lb)
                            ce = _check_point(x_flat, tag=f"grid_dim<=4_{coeffs}")
                            if ce is not None:
                                return ce

                    # -------- 3) 中维：per-dim 边界点（d<=16）--------
                    # 从中心出发，逐维推到 lb_i / ub_i
                    if dim <= 16:
                        center = 0.5 * (lb + ub)
                        for i in range(dim):
                            # lb 方向
                            x_flat = center.clone()
                            x_flat[i] = lb[i]
                            ce = _check_point(x_flat, tag=f"per_dim_lb_i={i}")
                            if ce is not None:
                                return ce
                            # ub 方向
                            x_flat = center.clone()
                            x_flat[i] = ub[i]
                            ce = _check_point(x_flat, tag=f"per_dim_ub_i={i}")
                            if ce is not None:
                                return ce

                    # -------- 4) 随机点：uniform in [lb, ub] --------
                    # 为了不太慢，这里采样 max_random 个
                    for k in range(max_random):
                        r = torch.rand_like(lb)
                        x_flat = lb + r * (ub - lb)
                        ce = _check_point(x_flat, tag=f"random_spec[{k}]")
                        if ce is not None:
                            return ce

                    # 如果基于 spec 的 sampling 也没找到，就退回老逻辑
                    logger.info(
                        "  [CE search] No CE found from INPUT_SPEC-guided sampling, "
                        "falling back to legacy test_cases."
                    )
        except Exception as e:
            logger.warning(
                "  [CE search] INPUT_SPEC-guided sampling failed (%s), "
                "falling back to legacy test_cases.",
                e,
            )

        # ============================================================
        # 2. 退回原来的 center / boundary / random 逻辑，但加强 random 次数
        # ============================================================
        # 先跑一次 center / boundary
        legacy_cases = ['center', 'boundary']
        for test_case in legacy_cases:
            input_tensor = self.factory.generate_test_input(name, test_case)
            input_tensor = input_tensor.to(device=self.device, dtype=self.dtype)
            results = model(input_tensor)

            if isinstance(results, dict):
                if results.get('input_satisfied', False) and not results.get('output_satisfied', True):
                    logger.info(f"  🔴 Counterexample found in legacy '{test_case}' test case")
                    logger.info(f"     Input explanation: {results.get('input_explanation')}")
                    logger.info(f"     Output explanation: {results.get('output_explanation')}")
                    return (input_tensor, results)

        # 再多跑几次 random（之前只跑 1 次，这里跑 max_random 次）
        for k in range(max_random):
            input_tensor = self.factory.generate_test_input(name, 'random')
            input_tensor = input_tensor.to(device=self.device, dtype=self.dtype)
            results = model(input_tensor)

            if isinstance(results, dict):
                if results.get('input_satisfied', False) and not results.get('output_satisfied', True):
                    logger.info(f"  🔴 Counterexample found in legacy random[{k}]")
                    logger.info(f"     Input explanation: {results.get('input_explanation')}")
                    logger.info(f"     Output explanation: {results.get('output_explanation')}")
                    return (input_tensor, results)

        # 还是没找到，就老老实实返回 None
        return None

    
    def validate_counterexamples(
        self, 
        networks: Optional[List[str]] = None,
        solvers: List[str] = ['gurobi', 'torchlp']
    ) -> Dict[str, Any]:
        """
        Level 1: Validate verifier soundness using concrete counterexamples.
        
        Args:
            networks: List of network names (None = all networks)
            solvers: List of solver names to test
            
        Returns:
            Summary dictionary with validation results
        """
        if networks is None:
            networks = self.factory.list_networks()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"LEVEL 1: COUNTEREXAMPLE/SOUNDNESS VALIDATION")
        logger.info(f"{'='*80}")
        logger.info(f"Testing {len(networks)} networks with {len(solvers)} solvers")
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")
        logger.info(f"{'='*80}\n")
        
        for network in networks:
            for solver in solvers:
                try:
                    self._validate_counterexample_single(network, solver)
                except Exception as e:
                    logger.error(f"Validation failed for {network}/{solver}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Add error result if not already added
                    error_result = {
                        'network': network,
                        'solver': solver,
                        'validation_type': 'counterexample',
                        'status': 'ERROR',
                        'error': f"Outer exception: {str(e)}",
                        'concrete_counterexample': False
                    }
                    self.validation_results.append(error_result)
        
        return self._compute_summary(validation_type='counterexample')
    
    def _validate_counterexample_single(
        self, 
        name: str, 
        solver: str
    ) -> Dict[str, Any]:
        """
        Validate verifier correctness for a single network (Level 1).
        
        Args:
            name: Network name from examples_config.yaml
            solver: 'gurobi' or 'torchlp'
            
        Returns:
            Validation result dictionary with status and details
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Validating: {name} (solver: {solver})")
        logger.info(f"{'='*80}")
        
        solver_ce_results: Optional[Dict[str, Any]] = None
        
        # Step 1: Load ACT Net from factory
        act_net = self.factory.get_act_net(name)
        # Debug: check input var wiring
        try:
            inp_layer = next((L for L in act_net.layers if getattr(L, "kind", "") == "INPUT"), None)
            if inp_layer:
                logger.debug(f"  [ACT INPUT] vars={len(inp_layer.out_vars)} shape={inp_layer.meta.get('shape')}")
            specs = gather_input_spec_layers(act_net)
            seed = seed_from_input_specs(specs)
            input_ids = get_input_ids(act_net)
            logger.debug(f"  [ACT INPUT_SPEC] ids={len(input_ids)}, seed_lb shape={tuple(seed.lb.shape)}, "
                         f"lb[min,max]=[{seed.lb.min():.4f},{seed.lb.max():.4f}], "
                         f"ub[min,max]=[{seed.ub.min():.4f},{seed.ub.max():.4f}]")
        except Exception:
            pass
        # # print each layer info for debugging
        # for layer in act_net.layers:
        #     layer_name = layer.meta.get("name", "unnamed")
        #     logger.error(f"  [ACT Layer] ID: {layer.id}, Kind: {layer.kind}, Name: {layer_name}")
        
        # Step 2: Create PyTorch model for concrete execution
        model = self.factory.create_model(name, load_weights=True)
        model = model.to(device=self.device, dtype=self.dtype)
        # counterexample = self.find_concrete_counterexample(name, model)
        counterexample = self.find_concrete_counterexample(
            name=name,
            model=model,
            act_net=act_net,
        )
        
        # Step 3: Run formal verifier on ACT Net
        logger.info(f"\n  🔍 Running formal verifier ({solver})...")
        
        try:
            if solver == 'gurobi':
                try:
                    solver_instance = GurobiSolver()
                except Exception as e:
                    if "License" in str(e):
                        logger.warning(f"     Skipping gurobi for {name}: {e}")
                        skip_result = {
                            'network': name,
                            'solver': solver,
                            'validation_type': 'counterexample',
                            'validation_status': 'INCONCLUSIVE',
                            'concrete_counterexample': counterexample is not None,
                            'verifier_result': 'ERROR',
                            'explanation': f"Skipped due to Gurobi license issue: {e}"
                        }
                        self.validation_results.append(skip_result)
                        return skip_result
                    raise
            elif solver == 'torchlp':
                solver_instance = TorchLPSolver()
            else:
                raise ValueError(f"Unknown solver: {solver}")
            
            verify_result = verify_once(act_net, solver=solver_instance)
            verifier_status = verify_result.status
            logger.info(f"     Verifier result: {verifier_status}")
            
            solver_ce_results = None

            # If verifier found counterexample, validate it with model
            if verify_result.counterexample is not None:
                logger.info(f"     Verifier counterexample shape: {verify_result.counterexample.shape}")
                # Reshape CE to the model's expected input shape (avoid conv2d shape errors)
                ce_raw = verify_result.counterexample
                input_shape = None
                for layer in act_net.layers:
                    if getattr(layer, "kind", None) == "INPUT":
                        input_shape = layer.meta.get("shape")
                        break
                try:
                    if input_shape is not None:
                        ce_tensor = ce_raw.view(*input_shape)
                    else:
                        ce_tensor = ce_raw.unsqueeze(0)
                except Exception as reshape_err:
                    logger.warning(f"     CE reshape failed, using vector: {reshape_err}")
                    ce_tensor = ce_raw.unsqueeze(0)
                ce_tensor = ce_tensor.to(device=self.device, dtype=self.dtype)
                ce_results = model(ce_tensor)
                if isinstance(ce_results, dict):
                    solver_ce_results = ce_results
                    logger.info(
                        "     CE validation: input_sat=%s, output_sat=%s",
                        ce_results.get("input_satisfied", None),
                        ce_results.get("output_satisfied", None),
                    )
                    # Optional debug: fixed-input LP check using solver CE
                    if os.environ.get("ACT_DEBUG_FIXED_INPUT_LP") and ce_results.get("output") is not None:
                        try:
                            from act.back_end.verifier import debug_fixed_input_lp, get_assert_layer
                            assert_layer = get_assert_layer(act_net)
                            t_idx = int(assert_layer.meta.get("y_true", 0))
                            out_tensor = ce_results["output"]
                            if out_tensor.dim() > 1:
                                out_tensor = out_tensor.view(-1)
                            j_alt = int(out_tensor.argmax().item())
                            # Fresh solver instance of same class as solver_instance
                            debug_solver = solver_instance.__class__()
                            st_dbg, ce_dbg, stats_dbg = debug_fixed_input_lp(
                                net=act_net,
                                solver=debug_solver,
                                x_ce=ce_tensor.detach().cpu(),
                                t=t_idx,
                                j_alt=j_alt,
                                timelimit=None,
                            )
                            logger.info(
                                "     [DEBUG FIXED LP] status=%s, ce_exists=%s, stats=%s",
                                st_dbg,
                                ce_dbg is not None,
                                stats_dbg,
                            )
                        except Exception as dbg_e:
                            logger.warning("     [DEBUG FIXED LP] failed: %s", dbg_e)
                else:
                    logger.warning("     CE validation returned unexpected result type; cannot interpret as spec/property.")
            else:
                # Even if CE was filtered out inside verify_once, we may still
                # have checker results stored in stats (ce_checks). Use them to
                # distinguish spurious vs real solver CEs for non-class specs.
                ce_checks = verify_result.stats.get("ce_checks", None)
                if isinstance(ce_checks, dict):
                    solver_ce_results = {
                        "input_satisfied": ce_checks.get("input_sat"),
                        "output_satisfied": not bool(ce_checks.get("output_violated", False)),
                    }
                    logger.info(
                        "     CE validation (from stats): input_sat=%s, output_sat=%s",
                        solver_ce_results.get("input_satisfied", None),
                        solver_ce_results.get("output_satisfied", None),
                    )
            
        except Exception as e:
            # Handle known license issues gracefully
            if solver == 'gurobi' and "License" in str(e):
                logger.warning(f"     Skipping gurobi for {name}: {e}")
                skip_result = {
                    'network': name,
                    'solver': solver,
                    'validation_type': 'counterexample',
                    'validation_status': 'INCONCLUSIVE',
                    'concrete_counterexample': counterexample is not None,
                    'verifier_result': 'ERROR',
                    'explanation': f"Skipped due to Gurobi license issue: {e}"
                }
                self.validation_results.append(skip_result)
                return skip_result

            logger.error(f"     Verifier failed: {e}")
            import traceback
            traceback.print_exc()
            error_result = {
                'network': name,
                'solver': solver,
                'validation_type': 'counterexample',
                'status': 'ERROR',
                'error': str(e),
                'concrete_counterexample': counterexample is not None
            }
            self.validation_results.append(error_result)
            return error_result
        
        # === DEBUG: 针对 reachability_tight + gurobi 打印 RANGE 违例度对比 ===
        try:
            if name == "reachability_tight" and solver == "gurobi":
                self._debug_range_mismatch(
                    name=name,
                    act_net=act_net,
                    model=model,
                    verify_result=verify_result,
                )
        except Exception as dbg_e:
            logger.warning("  [DEBUG RANGE] 调试失败: %s", dbg_e)
        # =====================================================================

        
        # Step 4: Cross-validate results
        validation = self._cross_validate_counterexample(
            network_name=name,
            solver_name=solver,
            concrete_counterexample=counterexample,
            verifier_status=verifier_status,
            solver_ce_results=solver_ce_results,
        )
        
        self.validation_results.append(validation)
        return validation
    
    def _debug_range_mismatch(
        self,
        name: str,
        act_net,
        model: torch.nn.Module,
        verify_result,
    ) -> None:
        """
        专门给 reachability_* 这类 RANGE 性质调试用：
        - 打印 solver CE 的 input 向量 (raw_ce_input)
        - 用 PyTorch VerifiableModel 重算一次 output
        - 计算 RANGE 违例度（PyTorch 视角）
        - 对比 solver 的 violation_var
        """
        try:
            assert_layer = get_assert_layer(act_net)
            if assert_layer.meta.get("kind") != OutKind.RANGE:
                return


            stats = verify_result.stats or {}
            raw_ce = stats.get("raw_ce_input", None)
            v_val = stats.get("violation_var", None)

            logger.error("=== DEBUG RANGE MISMATCH for %s ===", name)
            logger.error("  solver violation_var v = %r", v_val)

            if raw_ce is None:
                logger.error("  raw_ce_input is None (solver CE 未记录)，无法对齐调试。")
                logger.error("=== END DEBUG RANGE MISMATCH ===")
                return

            import numpy as np

            # 1) 打印 CE input（有限截断一下避免太长）
            raw_ce_np = np.asarray(raw_ce, dtype=float).reshape(-1)
            logger.error("  solver CE input (flat) shape=%s", raw_ce_np.shape)
            logger.error("  solver CE input (first 20 dims)=%s", raw_ce_np[:20])

            # 2) 把 CE 映射成 VerifiableModel 的输入形状
            input_shape = None
            for layer in act_net.layers:
                if getattr(layer, "kind", None) == "INPUT":
                    input_shape = layer.meta.get("shape")
                    break

            x = torch.from_numpy(raw_ce_np)
            if input_shape is not None:
                try:
                    x = x.view(*input_shape)
                except Exception as e:
                    logger.error(
                        "  reshape raw_ce -> %s 失败 (%s)，退化成 batch=1, flat 向量。",
                        input_shape, e,
                    )
                    x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(0)

            x = x.to(device=self.device, dtype=self.dtype)

            # 3) 用 PyTorch VerifiableModel 重算一次输出
            with torch.no_grad():
                res = model(x)

            if not isinstance(res, dict) or "output" not in res:
                logger.error("  model(x) 返回类型=%r，不含 'output'，无法进一步调试。", type(res))
                logger.error("=== END DEBUG RANGE MISMATCH ===")
                return

            y = res["output"].detach().cpu().numpy().reshape(-1)
            logger.error("  PyTorch output y (flat)=%s", y)

            # 4) 取 ASSERT 的 lb / ub，并计算 PyTorch 视角的违例度
            lb_t = assert_layer.params.get("lb", None)
            ub_t = assert_layer.params.get("ub", None)

            lb_np = None
            ub_np = None
            if lb_t is not None:
                lb_np = lb_t.detach().cpu().numpy().reshape(-1)
                logger.error("  RANGE lb=%s", lb_np)
            if ub_t is not None:
                ub_np = ub_t.detach().cpu().numpy().reshape(-1)
                logger.error("  RANGE ub=%s", ub_np)

            viol_terms = []

            if lb_np is not None:
                viol_lower = lb_np - y          # >0 表示 y < lb 的违例程度
                max_vl = float(viol_lower.max())
                viol_terms.append(max_vl)
                logger.error("  max(lb - y) = %.6f", max_vl)

            if ub_np is not None:
                viol_upper = y - ub_np          # >0 表示 y > ub 的违例程度
                max_vu = float(viol_upper.max())
                viol_terms.append(max_vu)
                logger.error("  max(y - ub) = %.6f", max_vu)

            if viol_terms:
                max_viol_py = float(max(viol_terms))
            else:
                max_viol_py = float("nan")

            logger.error("  PyTorch 视角 max RANGE violation = %.6f", max_viol_py)
            logger.error("  Solver 视角 violation_var v    = %r", v_val)
            logger.error("=== END DEBUG RANGE MISMATCH ===")

        except Exception as e:
            logger.exception(
                "Error in _debug_range_mismatch for %s: %s", name, e
            )

    
    def _cross_validate_counterexample(
        self,
        network_name: str,
        solver_name: str,
        concrete_counterexample: Optional[Tuple],
        verifier_status: str,
        solver_ce_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Cross-validate concrete inference vs formal verification (Level 1).

        更保守、不会误报 reachability_* 为 FAILED 的版本：

        规则：
        1. 如果存在“被确认的真实 CE”（测试样本 or 经过 checker 验证的 solver CE），
           则：
           - Verifier 返回 CERTIFIED → 这是确定的 soundness bug → FAILED
           - Verifier 返回 FALSIFIED  → 正常，认为 PASSED
           - Verifier 返回 UNKNOWN    → 不完整但 sound → ACCEPTABLE

        2. 如果目前没有任何被确认的 CE（测试没找到，solver CE 也没被确认为真），
           无论 verifier 给什么结果，都记为 INCONCLUSIVE，
           不再因为 FALSIFIED + “ce_checks 看起来像伪 CE”就报 FAILED。
        """
        result: Dict[str, Any] = {
            'network': network_name,
            'solver': solver_name,
            'validation_type': 'counterexample',
            'concrete_counterexample': concrete_counterexample is not None,
            'verifier_result': verifier_status,
            'validation_status': None,
            'explanation': None,
            'solver_ce_checked': False,
            'solver_ce_is_ce': False,
            'solver_ce_spurious': False,
        }

        # 1) 来自 center/boundary/random 测试的 ground truth
        test_ce_exists = concrete_counterexample is not None

        # 2) 来自 solver CE（通过 stats 或直接 CE）的 ground truth
        solver_ce_is_ce = False
        solver_ce_spurious = False
        if isinstance(solver_ce_results, dict):
            result['solver_ce_checked'] = True
            ce_in_sat = bool(solver_ce_results.get('input_satisfied', False))
            ce_out_sat = bool(solver_ce_results.get('output_satisfied', True))

            # 真 CE：输入满足 spec，输出违反 property
            solver_ce_is_ce = ce_in_sat and (not ce_out_sat)
            # 伪 CE：输入不满足 spec 或者输出没有违反 property
            solver_ce_spurious = (not ce_in_sat) or ce_out_sat

            result['solver_ce_is_ce'] = solver_ce_is_ce
            result['solver_ce_spurious'] = solver_ce_spurious

        # 是否存在任何“被确认真实”的 CE
        any_ce_exists = test_ce_exists or solver_ce_is_ce

        # ===============================
        # Case A: 至少有一个真实 CE
        # ===============================
        if any_ce_exists:
            if verifier_status == 'CERTIFIED':
                # 典型 false negative：真实有 CE，verifier 却说 CERTIFIED
                result['validation_status'] = 'FAILED'
                result['explanation'] = (
                    "🚨 SOUNDNESS BUG DETECTED! Verifier returned CERTIFIED but a "
                    "concrete counterexample exists (false negative)."
                )
                logger.error("\n  %s", result['explanation'])

            elif verifier_status == 'FALSIFIED':
                # 有真实 CE，verifier 也报告 FALSIFIED：方向正确
                result['validation_status'] = 'PASSED'
                result['explanation'] = (
                    "✅ CORRECT - Verifier reported FALSIFIED and at least one "
                    "real counterexample exists (from tests and/or solver CE)."
                )
                logger.info("\n  %s", result['explanation'])

            elif verifier_status == 'UNKNOWN':
                # 有真实 CE，但 verifier 不敢 CERTIFIED：不完整但 sound
                result['validation_status'] = 'ACCEPTABLE'
                result['explanation'] = (
                    "⚠️ INCOMPLETE - Verifier returned UNKNOWN while a concrete "
                    "counterexample exists (sound but incomplete)."
                )
                logger.warning("\n  %s", result['explanation'])

            else:
                result['validation_status'] = 'UNKNOWN'
                result['explanation'] = f"Unknown verifier result: {verifier_status!r}"
                logger.warning("\n  %s", result['explanation'])

            return result

        # ===============================
        # Case B: 当前没有任何被确认的 CE
        # ===============================
        # 测试没找到 CE，solver 给的 CE 也没被确认为真（可能是伪的，也可能我们检查不够强），
        # 这种情况我们只能说 “INCONCLUSIVE”，不能直接指控 verifier unsound。
        result['validation_status'] = 'INCONCLUSIVE'
        result['explanation'] = (
            "⚪ INCONCLUSIVE - No concrete counterexample found in testing, "
            "and solver-proposed CEs are not validated as real counterexamples. "
            f"Verifier result: {verifier_status}."
        )
        logger.info("\n  %s", result['explanation'])

        return result
    
    def validate_bounds(
        self,
        networks: Optional[List[str]] = None,
        tf_modes: List[str] = ['interval'],
        num_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Level 2: Validate abstract bounds overapproximate concrete values.
        
        Args:
            networks: List of network names (None = all networks)
            tf_modes: Transfer function modes to test ('interval', 'hybridz')
            num_samples: Number of concrete inputs to sample per network
            
        Returns:
            Summary dictionary with validation results
        """
        if networks is None:
            networks = self.factory.list_networks()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"LEVEL 2: BOUNDS/NUMERICAL VALIDATION")
        logger.info(f"{'='*80}")
        logger.info(f"Testing {len(networks)} networks with {len(tf_modes)} TF modes")
        logger.info(f"Samples per network: {num_samples}")
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")
        logger.info(f"{'='*80}\n")
        
        for network in networks:
            for tf_mode in tf_modes:
                try:
                    self._validate_bounds_single(network, tf_mode, num_samples)
                except Exception as e:
                    logger.error(f"Bounds validation failed for {network}/{tf_mode}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Add error result if not already added
                    error_result = {
                        'network': network,
                        'tf_mode': tf_mode,
                        'validation_type': 'bounds',
                        'status': 'ERROR',
                        'error': f"Outer exception: {str(e)}",
                        'samples_processed': 0
                    }
                    self.validation_results.append(error_result)
        
        return self._compute_summary(validation_type='bounds')
    
    def _validate_bounds_single(
        self,
        name: str,
        tf_mode: str,
        num_samples: int
    ) -> Dict[str, Any]:
        """
        Validate bounds for a single network (Level 2).
        
        Args:
            name: Network name
            tf_mode: Transfer function mode ('interval' or 'hybridz')
            num_samples: Number of concrete inputs to sample
            
        Returns:
            Validation result dictionary
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Validating bounds: {name} (tf_mode: {tf_mode})")
        logger.info(f"{'='*80}")
        
        # Step 1: Load ACT Net and PyTorch model
        act_net = self.factory.get_act_net(name)
        model = self.factory.create_model(name, load_weights=True)
        model = model.to(device=self.device, dtype=self.dtype)
        
        # Step 2: Set transfer function mode globally
        from act.back_end.transfer_functions import set_transfer_function_mode
        set_transfer_function_mode(tf_mode)
        
        # Step 3: Sample concrete inputs
        violations: List[Dict[str, Any]] = []
        total_checks = 0
        
        def _get_input_bounds_from_act(act_net_inner):
            from act.back_end.core import Bounds
            for layer in act_net_inner.layers:
                if layer.kind != "INPUT_SPEC":
                    continue

                params = layer.params or {}
                meta = layer.meta or {}

                # 1) BOX 优先
                if "lb" in params and "ub" in params:
                    return Bounds(
                        lb=params["lb"].flatten().to(device=self.device, dtype=self.dtype),
                        ub=params["ub"].flatten().to(device=self.device, dtype=self.dtype),
                    )

                # 2) LINF_BALL: center + eps
                if "center" in params and "eps" in meta:
                    center = params["center"].flatten().to(device=self.device, dtype=self.dtype)
                    eps = meta["eps"]
                    if not torch.is_tensor(eps):
                        eps = torch.tensor(eps, device=self.device, dtype=self.dtype)
                    else:
                        eps = eps.to(device=self.device, dtype=self.dtype)
                    return Bounds(lb=center - eps, ub=center + eps)
            return None

        # def _get_input_bounds_from_act(act_net_inner):
        #     from act.back_end.core import Bounds
        #     for layer in act_net_inner.layers:
        #         if layer.kind == "INPUT_SPEC":
        #             params = layer.params or {}
        #             meta = layer.meta or {}
        #             if 'lb' in params and 'ub' in params:
        #                 return Bounds(
        #                     lb=params['lb'].flatten().to(device=self.device, dtype=self.dtype),
        #                     ub=params['ub'].flatten().to(device=self.device, dtype=self.dtype),
        #                 )
        #         # LINF_BALL: center in params, eps in meta
        #         if 'center' in params and 'eps' in meta:
        #             center = params['center'].flatten().to(device=self.device, dtype=self.dtype)
        #             eps = meta['eps']
        #             if not torch.is_tensor(eps):
        #                 eps = torch.tensor(eps, device=self.device, dtype=self.dtype)
        #             else:
        #                 eps = eps.to(device=self.device, dtype=self.dtype)
        #             return Bounds(lb=center - eps, ub=center + eps)
        #     return None

        
        # # Extract input bounds from ACT net (prefer INPUT_SPEC params)
        # def _get_input_bounds_from_act(act_net_inner):
        #     from act.back_end.core import Bounds
        #     for layer in act_net_inner.layers:
        #         if layer.kind == "INPUT_SPEC":
        #             params = layer.params or {}
        #             if 'lb' in params and 'ub' in params:
        #                 return Bounds(lb=params['lb'].flatten().to(device=self.device, dtype=self.dtype),
        #                               ub=params['ub'].flatten().to(device=self.device, dtype=self.dtype))
        #             if 'center' in params and 'eps' in params:
        #                 center = params['center'].flatten().to(device=self.device, dtype=self.dtype)
        #                 eps = params['eps'].to(device=self.device, dtype=self.dtype)
        #                 # eps may be scalar tensor
        #                 return Bounds(lb=center - eps, ub=center + eps)
        #     return None
        
        spec_bounds = _get_input_bounds_from_act(act_net)
        
        for sample_idx in range(num_samples):
            # Generate random input within spec
            input_tensor = self.factory.generate_test_input(name, 'random')
            input_tensor = input_tensor.to(device=self.device, dtype=self.dtype)
            
            # Step 4: Get concrete activations via forward hooks
            concrete_activations = self._get_concrete_activations(model, input_tensor, act_net)
            
            # Step 5: Prepare entry fact from input tensor
            from act.back_end.core import Fact, Bounds, ConSet
            # entry_id = 0  # INPUT layer is typically layer 0
            entry_id = find_entry_layer_id(act_net)
            if spec_bounds is not None:
                input_bounds = spec_bounds
            else:
                input_bounds = Bounds(lb=input_tensor.flatten(), ub=input_tensor.flatten())
            # Use an empty constraint set for inputs so downstream analysis
            # never iterates over a None cons field.
            entry_fact = Fact(bounds=input_bounds, cons=ConSet())
            
            # Step 6: Run abstract analysis
            try:
                before, after, globalC = analyze(act_net, entry_id, entry_fact)
                # logger.info(f"  [This net has {len(after)} layers].")
                
                # Step 7: Check bounds containment
                for layer_id, concrete_vals in concrete_activations.items():
                    # logger.error(f"  [Sample {sample_idx}] Checking layer {layer_id}...")   
                    if layer_id not in after:
                        continue
                    
                    abstract_bounds = after[layer_id].bounds
                    lb = abstract_bounds.lb
                    ub = abstract_bounds.ub
                    
                    # Flatten concrete values to match ACT's 1D representation
                    concrete_vals_flat = concrete_vals.flatten()
                    
                    # Ensure shapes match (ACT may have different neuron counts)
                    if concrete_vals_flat.shape != lb.shape:
                        logger.warning(
                            "  ⚠️ Shape mismatch at layer %s: concrete=%s, abstract=%s. Skipping.",
                            layer_id, tuple(concrete_vals_flat.shape), tuple(lb.shape),
                        )
                        continue
                    
                    # Check if concrete values are within bounds
                    violations_mask = (concrete_vals_flat < lb) | (concrete_vals_flat > ub)
                    num_violations = int(violations_mask.sum().item())
                    total_checks += int(concrete_vals_flat.numel())
                    
                    if num_violations > 0:
                        if name in ("control_strict", "reachability_tight") \
                           and layer_id == 0 and sample_idx == 0:
                            logger.error("=== DEBUG %s / tf_mode=%s / sample=%d / layer=%d ===",
                                         name, tf_mode, sample_idx, layer_id)
                            logger.error("concrete_vals_flat: %s", concrete_vals_flat)
                            logger.error("lb: %s", lb)
                            logger.error("ub: %s", ub)
                            logger.error("viol_lt_lb idx: %s",
                                         (concrete_vals_flat < lb).nonzero(as_tuple=False).view(-1))
                            logger.error("viol_gt_ub idx: %s",
                                         (concrete_vals_flat > ub).nonzero(as_tuple=False).view(-1))
                            logger.error(
                                "concrete[min, max]=[%.6f, %.6f], "
                                "lb[min, max]=[%.6f, %.6f], "
                                "ub[min, max]=[%.6f, %.6f]",
                                float(concrete_vals_flat.min().item()),
                                float(concrete_vals_flat.max().item()),
                                float(lb.min().item()), float(lb.max().item()),
                                float(ub.min().item()), float(ub.max().item()),
                            )
                            logger.error("=== END DEBUG ===")
                        violation_info = {
                            'sample_idx': sample_idx,
                            'layer_id': layer_id,
                            'num_violations': num_violations,
                            'total_neurons': int(concrete_vals_flat.numel()),
                            'concrete_min': float(concrete_vals_flat.min().item()),
                            'concrete_max': float(concrete_vals_flat.max().item()),
                            'abstract_lb': float(lb.min().item()),
                            'abstract_ub': float(ub.max().item()),
                        }
                        violations.append(violation_info)
                        logger.error(
                            "  ❌ Bounds violation at layer %s: %d/%d neurons",
                            layer_id, num_violations, int(concrete_vals_flat.numel()),
                        )
            
            except Exception as e:
                logger.error(f"  ⚠️ Abstract analysis failed for sample {sample_idx}: {e}")
                error_result = {
                    'network': name,
                    'tf_mode': tf_mode,
                    'validation_type': 'bounds',
                    'status': 'ERROR',
                    'error': str(e),
                    'samples_processed': sample_idx
                }
                self.validation_results.append(error_result)
                return error_result
        
        # Step 6: Summarize results
        if len(violations) > 0:
            result = {
                'network': name,
                'tf_mode': tf_mode,
                'validation_type': 'bounds',
                'validation_status': 'FAILED',
                'explanation': f"🚨 UNSOUND BOUNDS: {len(violations)} violations found across {num_samples} samples",
                'total_checks': total_checks,
                'violations': violations
            }
            logger.error(f"\n  {result['explanation']}")
        else:
            result = {
                'network': name,
                'tf_mode': tf_mode,
                'validation_type': 'bounds',
                'validation_status': 'PASSED',
                'explanation': f"✅ SOUND BOUNDS: All {total_checks} checks passed across {num_samples} samples",
                'total_checks': total_checks,
                'violations': []
            }
            logger.info(f"\n  {result['explanation']}")
        
        self.validation_results.append(result)
        return result
    
    def _get_concrete_activations(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        act_net=None
    ) -> Dict[int, torch.Tensor]:
        """
        Get concrete activation values by running forward pass with hooks.
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor
            act_net: Optional ACT Net to align hooks to ACT layer ids
            
        Returns:
            Dictionary mapping layer_id to activation tensor
        """
        activations: Dict[int, torch.Tensor] = {}
        hooks: List[Any] = []
        collected: List[torch.Tensor] = []
        
        def make_hook(temp_id: int):
            def hook(module, input, output):
                collected.append(output.detach().clone())
            return hook
        
        hook_kinds = {
            "DENSE": torch.nn.Linear,
            "CONV2D": torch.nn.Conv2d,
            "RELU": torch.nn.ReLU,
            "FLATTEN": torch.nn.Flatten,
        }
        
        # Register hooks on relevant torch modules; map to ACT ids after forward
        temp_id = 0
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ReLU, torch.nn.Flatten)):
                hook = module.register_forward_hook(make_hook(temp_id))
                hooks.append(hook)
                temp_id += 1
        
        # Run forward pass
        with torch.no_grad():
            model(input_tensor)
        
        # Align collected activations with ACT layer ids so shapes match
        if act_net is not None:
            act_ids = [layer.id for layer in act_net.layers if layer.kind in hook_kinds]
            if len(act_ids) != len(collected):
                logger.warning(
                    "  ⚠️ Hook count mismatch: torch collected %d, ACT hookable layers=%d; aligning by position.",
                    len(collected), len(act_ids),
                )
            for idx, act_id in enumerate(act_ids[:len(collected)]):
                activations[act_id] = collected[idx]
        else:
            for idx, tensor in enumerate(collected):
                activations[idx] = tensor
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def validate_comprehensive(
        self,
        networks: Optional[List[str]] = None,
        solvers: List[str] = ['gurobi', 'torchlp'],
        tf_modes: List[str] = ['interval'],
        num_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Run both Level 1 and Level 2 validations.
        
        Args:
            networks: List of network names (None = all networks)
            solvers: List of solver names for Level 1
            tf_modes: Transfer function modes for Level 2
            num_samples: Number of samples for Level 2
            
        Returns:
            Combined summary dictionary
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPREHENSIVE VERIFICATION VALIDATION")
        logger.info(f"{'='*80}")
        logger.info(f"Running both Level 1 (Counterexample) and Level 2 (Bounds) validation")
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")
        logger.info(f"{'='*80}\n")
        
        # Run Level 1
        summary_l1 = self.validate_counterexamples(networks=networks, solvers=solvers)
        
        # Run Level 2
        summary_l3 = self.validate_bounds(networks=networks, tf_modes=tf_modes, num_samples=num_samples)
        
        # Combine summaries - FAILED if any failures OR errors
        has_failures = (summary_l1.get('failed', 0) > 0 or summary_l3.get('failed', 0) > 0)
        has_errors = (summary_l1.get('errors', 0) > 0 or summary_l3.get('errors', 0) > 0)
        
        if has_failures:
            overall_status = 'FAILED'  # Critical: verifier is unsound
        elif has_errors:
            overall_status = 'ERROR'   # Backend bugs prevent validation
        else:
            overall_status = 'PASSED'  # All tests passed
        
        combined = {
            'level1_counterexample': summary_l1,
            'level3_bounds': summary_l3,
            'overall_status': overall_status
        }
        
        self._print_comprehensive_summary(combined)
        return combined
    
    def _compute_summary(self, validation_type: str) -> Dict[str, Any]:
        """
        Compute validation summary statistics for specific validation type.
        
        Args:
            validation_type: 'counterexample' or 'bounds'
        """
        results = [r for r in self.validation_results if r.get('validation_type') == validation_type]
        total = len(results)
        
        if total == 0:
            return {
                'validation_type': validation_type,
                'total': 0,
                'passed': 0,
                'failed': 0,
                'acceptable': 0,
                'inconclusive': 0,
                'errors': 0,
                'results': [],
                'error_message': 'No validation results (all tests encountered errors)'
            }
        
        passed = sum(1 for r in results if r.get('validation_status') == 'PASSED')
        failed = sum(1 for r in results if r.get('validation_status') == 'FAILED')
        acceptable = sum(1 for r in results if r.get('validation_status') == 'ACCEPTABLE')
        inconclusive = sum(1 for r in results if r.get('validation_status') == 'INCONCLUSIVE')
        errors = sum(1 for r in results if r.get('status') == 'ERROR')
        
        summary: Dict[str, Any] = {
            'validation_type': validation_type,
            'total': total,
            'passed': passed,
            'failed': failed,
            'acceptable': acceptable,
            'inconclusive': inconclusive,
            'errors': errors,
            'results': results
        }
        
        if validation_type == 'counterexample':
            summary['counterexamples_found'] = sum(1 for r in results if r.get('concrete_counterexample', False))
            summary['critical_bugs'] = failed
        elif validation_type == 'bounds':
            summary['total_checks'] = sum(r.get('total_checks', 0) for r in results)
            summary['total_violations'] = sum(len(r.get('violations', [])) for r in results)
        
        self._print_summary(summary)
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print validation summary for specific validation type."""
        validation_type = summary.get('validation_type', 'unknown')
        
        print("\n" + "="*80)
        print(f"VALIDATION SUMMARY - {validation_type.upper()}")
        print("="*80)
        
        if summary['total'] == 0:
            print()
            print("⚠️  No validation tests completed successfully")
            if 'error_message' in summary:
                print(f"   {summary['error_message']}")
            print("="*80)
            return
        
        print(f"\nTotal validation tests: {summary['total']}")
        
        if validation_type == 'counterexample':
            print(f"Concrete counterexamples found: {summary.get('counterexamples_found', 0)}")
        elif validation_type == 'bounds':
            print(f"Total bound checks: {summary.get('total_checks', 0)}")
            print(f"Total violations: {summary.get('total_violations', 0)}")
        
        print()
        print(f"✅ PASSED:       {summary['passed']}")
        if validation_type == 'counterexample':
            print(f"⚠️  ACCEPTABLE:   {summary['acceptable']}")
            print(f"⚪ INCONCLUSIVE: {summary['inconclusive']}")
        print(f"❌ ERRORS:       {summary['errors']}")
        print(f"🚨 FAILED:       {summary['failed']}")
        print("="*80)
        
        if summary['failed'] > 0:
            print(f"\n🚨 CRITICAL: {validation_type.upper()} validation failed!")
            if validation_type == 'counterexample':
                print("Soundness bugs detected in the following networks:")
            else:
                print("Unsound bounds detected in the following networks:")
            for result in summary['results']:
                if result.get('validation_status') == 'FAILED':
                    if validation_type == 'counterexample':
                        print(f"  - {result['network']} ({result['solver']})")
                    else:
                        print(f"  - {result['network']} ({result['tf_mode']})")
            print()
        elif summary['errors'] > 0:
            print(f"\n⚠️  All {validation_type} validation tests encountered errors!")
            print("This indicates pre-existing bugs in the verification backend.")
            print()
        else:
            print(f"\n✅ {validation_type.upper()} validation PASSED!")
        
        print("="*80)
    
    def _print_comprehensive_summary(self, combined: Dict[str, Any]):
        """Print comprehensive summary for both validation levels."""
        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION SUMMARY")
        print("="*80)
        
        l1 = combined['level1_counterexample']
        l3 = combined['level3_bounds']
        
        print(f"\nLevel 1 (Counterexample): {l1['passed']}/{l1['total']} passed, {l1['failed']} failed, {l1['errors']} errors")
        print(f"Level 2 (Bounds):         {l3['passed']}/{l3['total']} passed, {l3['failed']} failed, {l3['errors']} errors")
        print()
        print(f"Overall Status: {combined['overall_status']}")
        print("="*80)


def main():
    """Run verification validation test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ACT Verification Validator')
    parser.add_argument('--mode', choices=['counterexample', 'bounds', 'comprehensive'],
                       default='comprehensive', help='Validation mode')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--dtype', default='float64', choices=['float32', 'float64'],
                       help='Data type')
    parser.add_argument('--networks', nargs='+', help='Specific networks to test')
    parser.add_argument('--solvers', nargs='+', default=['gurobi', 'torchlp'],
                       help='Solvers for Level 1')
    parser.add_argument('--tf-modes', nargs='+', default=['interval'],
                       help='Transfer function modes for Level 2')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of samples for Level 2')
    parser.add_argument('--ignore-errors', action='store_true',
                       help='Always exit 0 (ignore failures and errors for CI)')
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype = torch.float64 if args.dtype == 'float64' else torch.float32
    
    # Create validator
    validator = VerificationValidator(device=args.device, dtype=dtype)
    
    # Run validation
    if args.mode == 'counterexample':
        summary = validator.validate_counterexamples(
            networks=args.networks,
            solvers=args.solvers
        )
        # Exit 1 if any failures OR errors detected
        exit_code = 1 if (summary['failed'] > 0 or summary['errors'] > 0) else 0
    elif args.mode == 'bounds':
        summary = validator.validate_bounds(
            networks=args.networks,
            tf_modes=args.tf_modes,
            num_samples=args.samples
        )
        # Exit 1 if any failures OR errors detected
        exit_code = 1 if (summary['failed'] > 0 or summary['errors'] > 0) else 0
    else:  # comprehensive
        combined = validator.validate_comprehensive(
            networks=args.networks,
            solvers=args.solvers,
            tf_modes=args.tf_modes,
            num_samples=args.samples
        )
        # Exit 1 for both FAILED (verification bugs) and ERROR (backend bugs)
        exit_code = 1 if combined['overall_status'] in ['FAILED', 'ERROR'] else 0
    
    # Override exit code if --ignore-errors is set
    if args.ignore_errors:
        exit_code = 0
    
    # Print debug file location (GUARDED)
    if PerformanceOptions.debug_tf:
        logger.info(f"\n📝 Debug log written to: {PerformanceOptions.debug_output_file}")
    
    return exit_code


if __name__ == "__main__":
    import sys
    sys.exit(main())
