
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
##   Abstract Constraint Transformer (ACT) - αβ-CROWN Integration            ##
##                                                                           ##
##   doctormeeee (https://github.com/doctormeeee) and contributors           ##
##   Copyright (C) 2024-2025                                                 ##
##                                                                           ##
##   This file contains code adapted from the open-source αβ-CROWN           ##
##   project with modifications and enhancements for integration             ##
##   within the ACT verification framework.                                  ##
##                                                                           ##
##   Original αβ-CROWN source:                                               ##
##   - Repository: https://github.com/Verified-Intelligence/alpha-beta-CROWN ##
##     Copyright (C) 2021-2025 The α,β-CROWN Team                            ##
##     Licensed under BSD 3-Clause License                                   ##
##     Primary authors: Huan Zhang, Zhouxing Shi, Xiangru Zhong              ##
##                                                                           ##
##   ACT modifications include:                                              ##
##   - Integration with ACT unified interface and parameter system           ##
##   - Compatibility improvements for ACT verification workflows             ##
##                                                                           ##
##   This integration enables seamless αβ-CROWN verification within          ##
##   the ACT framework while maintaining full compatibility with             ##
##   original αβ-CROWN capabilities and performance characteristics.         ##
##                                                                           ##
###############################################################################

import copy
import socket
import random
import os
import sys
import time
import gc
import torch
import numpy as np
from collections import defaultdict
from act.util.path_config import get_abcrown_path

# Apply MNIST mirrors early: yann.lecun.com is no longer reliably reachable.
# Must run before any abcrown data-loading code touches torchvision.
from act.util.torchvision_urls import configure_mirror_urls
configure_mirror_urls()

# Import ABCrown modules from external library
sys.path.insert(0, os.path.join(get_abcrown_path(), 'complete_verifier'))
import arguments
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import stop_criterion_all, stop_criterion_batch_any
from auto_LiRPA.operators.convolution import BoundConv
from jit_precompile import precompile_jit_kernels
from beta_CROWN_solver import LiRPANet
from lp_mip_solver import mip, check_enable_refinement
from attack import attack
from attack.attack_pgd import check_and_save_cex
from utils import Logger, print_model
from specifications import (trim_batch, batch_vnnlib, sort_targets,
                            prune_by_idx, add_rhs_offset)
from loading import load_model_and_vnnlib, parse_run_mode, adhoc_tuning, Customized
from bab import general_bab
from input_split.batch_branch_and_bound import input_bab_parallel
from read_vnnlib import read_vnnlib
from cuts.cut_utils import terminate_mip_processes, terminate_mip_processes_by_c_matching
from lp_test import compare_optimized_bounds_against_lp_bounds

class abCrown:

    def __init__(self, method=None, args=None):

        self.method = method
        
        # Convert ACT long format spec types to abCrown short format for external compatibility
        converted_args = []
        i = 0
        while i < len(args):
            if args[i] == '--spec_type' and i + 1 < len(args):
                converted_args.append('--spec_type')
                # Map long format to short format for external abCrown compatibility
                spec_type = args[i + 1]
                if spec_type == 'local_lp':
                    converted_args.append('lp')
                elif spec_type == 'set_box':
                    converted_args.append('box')
                elif spec_type in ['set_vnnlib', 'local_vnnlib']:
                    converted_args.append('bound')
                else:
                    converted_args.append(spec_type)  # Keep as is if unknown
                i += 2
            else:
                converted_args.append(args[i])
                i += 1
        
        arguments.Config.parse_config(converted_args)

    def settings(self):

        print(f'Experiments at {time.ctime()} on {socket.gethostname()}')
        torch.manual_seed(arguments.Config['general']['seed'])
        random.seed(arguments.Config['general']['seed'])
        np.random.seed(arguments.Config['general']['seed'])
        torch.set_printoptions(precision=8)
        device = arguments.Config['general']['device']
        if device != 'cpu':
            torch.cuda.manual_seed_all(arguments.Config['general']['seed'])

            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        bab_args = arguments.Config['bab']
        debug_args = arguments.Config['debug']
        timeout_threshold = bab_args['timeout']
        select_instance = arguments.Config['data']['select_instance']
        complete_verifier = arguments.Config['general']['complete_verifier']

        if bab_args['backing_up_max_domain'] is None:
            arguments.Config['bab']['backing_up_max_domain'] = bab_args['initial_max_domains']

        print("Config.all_args:", arguments.Config.all_args)
        (run_mode, save_path, file_root, example_idx_list, model_ori,
        vnnlib_all, shape) = parse_run_mode()
        self.logger = Logger(run_mode, save_path, timeout_threshold)

        return run_mode, save_path, file_root, example_idx_list, model_ori, vnnlib_all, \
            shape, device, bab_args, debug_args, timeout_threshold, select_instance, complete_verifier

    def alpha_crown(
        self,
        model_ori,
        data,
        data_ub=None,
        data_lb=None,
        vnnlib=None,
        interm_bounds=None
    ):

        assert len(vnnlib) == 1, 'incomplete_verifier only support single x spec'
        input_x, specs = vnnlib[0]
        c_transposed = False

        if len(specs) > 1:

            assert all([len(_[0]) == 1 for _ in specs]), \
                'for each property in OR, only one clause supported so far'
            c = torch.concat([
                item[0] if isinstance(item[0], torch.Tensor) else torch.tensor(item[0])
                for item in specs], dim=0).unsqueeze(1).to(data)
            do_transpose = not arguments.Config['solver']['optimize_disjuncts_separately']
            rhs = torch.tensor(np.array([item[1] for item in specs])).to(data)
            if do_transpose and c.shape[0] != 1 and data.shape[0] == 1:
                c = c.transpose(0, 1)
                rhs = rhs.t()
                c_transposed = True
            else:
                if arguments.Config['solver']['prune_after_crown']:
                    raise NotImplementedError(
                        'To use optimize_disjuncts_separately=True, do not set '
                        'prune_after_crown=True'
                    )
            stop_func = stop_criterion_all(rhs)

        else:

            c = torch.tensor(specs[0][0]).unsqueeze(0).to(data)

            rhs = torch.tensor(specs[0][1], dtype=data.dtype, device=data.device).unsqueeze(0)
            stop_func = stop_criterion_batch_any(rhs)

        model = LiRPANet(model_ori, in_size=data.shape, c=c)

        bound_prop_method = arguments.Config['solver']['bound_prop_method']

        if isinstance(input_x, dict):

            ptb = PerturbationLpNorm(
                norm=input_x['norm'],
                eps=input_x['eps'], eps_min=input_x.get('eps_min', 0),
                x_L=data_lb, x_U=data_ub)
        else:
            norm = arguments.Config['specification']['norm']

            ptb = PerturbationLpNorm(norm=norm, x_L=data_lb, x_U=data_ub)
        x = BoundedTensor(data, ptb).to(data.device)
        output = model.net(x)
        print_model(model.net)
        print('Original output:', output)

        if arguments.Config['general']['save_output']:
            arguments.Globals['out']['pred'] = output.cpu()

        domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)

        global_lb, ret = model.build(
            domain, x, stop_criterion_func=stop_func, decision_thresh=rhs, vnnlib_ori=vnnlib,
            interm_bounds=interm_bounds)

        if c_transposed:

            global_lb = global_lb.t()
            rhs = rhs.t()

        if torch.any((global_lb - rhs) > 0, dim=-1).all():

            print('verified with init bound!')
            return 'safe-incomplete', {}, model

        if arguments.Config['attack']['pgd_order'] == 'middle':
            if ret['attack_images'] is not None:
                return 'unsafe-pgd', {}, model

        saved_alphas = defaultdict(dict)
        for m in model.net.optimizable_activations:
            for spec_name, alpha in m.alpha.items():

                saved_alphas[m.name][spec_name] = alpha.detach().clone()

        if bound_prop_method == 'alpha-crown':
            ret['activation_opt_params'] = {
                node.name: node.dump_optimized_params()
                for node in model.net.optimizable_activations
            }

        if c_transposed:
            ret['lower_bounds'][model.final_name] = ret['lower_bounds'][model.final_name].t()
            ret['upper_bounds'][model.final_name] = ret['upper_bounds'][model.final_name].t()
            if ret['lA'] is not None:
                ret['lA'] = {k: v.transpose(0, 1) for k, v in ret['lA'].items()}

        ret.update({'model': model, 'global_lb': global_lb, 'alpha': saved_alphas})
        return 'unknown', ret, model

    def beta_crown_bab(self, data_lb, data_ub, c, rhs,
            data=None, targets=None, vnnlib=None, timeout=None,
            time_stamp=0, data_dict=None, lower_bounds=None, upper_bounds=None,
            reference_alphas=None, attack_images=None, cplex_processes=None,
            activation_opt_params=None, reference_lA=None,
            model_incomplete=None, refined_betas=None,
            create_model=True, model=None, return_domains=False,
            max_iterations=None, property_idx=None, vnnlib_meta=None,
            orig_lirpa_model=None):

        if vnnlib_meta is None:
            vnnlib_meta = {
                'property_idx': 0, 'vnnlib_id': 0, 'benchmark_name': None
            }
        self.data_lb, self.data_ub, self.c, self.rhs = data_lb, data_ub, c, rhs
        self.data, self.targets, self.vnnlib = data, targets, vnnlib

        if create_model:
            self.model = LiRPANet(
                model, c=c, cplex_processes=cplex_processes,
                in_size=(data_lb.shape if len(targets) <= 1
                        else [len(targets)] + list(data_lb.shape[1:])),
                mip_building_proc=(orig_lirpa_model.mip_building_proc
                                   if orig_lirpa_model is not None else None)
            )
            if not model_incomplete:
                print_model(self.model.net)

        data_lb, data_ub = data_lb.to(self.model.device), data_ub.to(self.model.device)
        norm = arguments.Config['specification']['norm']
        if data_dict is not None:
            assert isinstance(data_dict['eps'], float)
            ptb = PerturbationLpNorm(
                norm=norm, eps=data_dict['eps'],
                eps_min=data_dict.get('eps_min', 0), x_L=data_lb, x_U=data_ub)
        else:
            ptb = PerturbationLpNorm(norm=norm, x_L=data_lb, x_U=data_ub)

        if data is not None:
            data = data.to(self.model.device)
            x = BoundedTensor(data, ptb).to(data_lb.device)
            output = self.model.net(x).flatten()
            print('Model prediction is:', output)

            if arguments.Config['general']['save_output']:
                arguments.Globals['out']['pred'] = output.cpu()

        else:
            x = BoundedTensor(data_lb, ptb).to(data_lb.device)

        self.domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)

        assert not return_domains, 'return_domains is only for input split for now'
        result = general_bab(
            self.model, self.domain, x,
            refined_lower_bounds=lower_bounds, refined_upper_bounds=upper_bounds,
            activation_opt_params=activation_opt_params, reference_lA=reference_lA,
            reference_alphas=reference_alphas, attack_images=attack_images,
            timeout=timeout, max_iterations=max_iterations,
            refined_betas=refined_betas, rhs=rhs, property_idx=property_idx,
            model_incomplete=model_incomplete, time_stamp=time_stamp)

        min_lb = result[0]
        if min_lb is None:
            min_lb = -torch.inf
        elif isinstance(min_lb, torch.Tensor):
            min_lb = min_lb.item()

        if result[2] == 'unsafe_bab':
            stats = result[3]
            adv_example = stats.counterexample.detach().to(self.model.device)
            verified_status, _ = check_and_save_cex(
                adv_example, self.model.net(adv_example), vnnlib,
                arguments.Config['attack']['cex_path'], 'unsafe')
            result = (min_lb, result[1], verified_status)
        else:
            result = (min_lb, *result[1:3])
        return result

    def beta_crown(
            self, model_ori, model_incomplete, vnnlib, batched_vnnlib, vnnlib_shape,
            index, timeout_threshold, bab_ret=None, cplex_processes=None,
            attack_images=None, attack_margins=None, results=None, vnnlib_id=None,
            benchmark_name=None, orig_lirpa_model=None
    ):
        start_time = time.time()

        enable_incomplete = arguments.Config['general']['enable_incomplete_verification']
        init_global_lb = results.get('global_lb', None)
        lower_bounds = results.get('lower_bounds', None)
        upper_bounds = results.get('upper_bounds', None)
        reference_alphas = results.get('alpha', None)
        lA = results.get('lA', None)
        cplex_cuts = (arguments.Config['bab']['cut']['enabled']
                    and arguments.Config['bab']['cut']['cplex_cuts'])
        bab_attack_enabled = arguments.Config['bab']['attack']['enabled']

        reference_alphas_cp = None
        if enable_incomplete:
            final_name = model_incomplete.final_name
            init_global_ub = upper_bounds[final_name]
            print('lA shape:', [lAitem.shape for lAitem in lA.values()])
            (batched_vnnlib, init_global_lb, init_global_ub,
            lA, attack_images) = sort_targets(
                batched_vnnlib, init_global_lb, init_global_ub,
                attack_images, attack_margins, results, model_incomplete)
            if reference_alphas is not None:
                reference_alphas_cp = copy.deepcopy(reference_alphas)

        solved_c_rows = []

        time_stamp = 0
        rhs_offsets = arguments.Config['specification']['rhs_offset']
        for property_idx, properties in enumerate(batched_vnnlib):

            print(f'\nProperties batch {property_idx}, size {len(properties[0])}')
            timeout = timeout_threshold - (time.time() - start_time)
            print(f'Remaining timeout: {timeout}')
            start_time_bab = time.time()
            print(f'Verifying property {property_idx} with {len(properties[0])} instances.')
            if arguments.Config['debug']['sanity_check'] == 'Full':
                rhs_offset = 0 if rhs_offsets is None else rhs_offsets[property_idx]
                timeout = timeout_threshold
                sanity_check_results = []
            else:
                rhs_offset = 0 if rhs_offsets is None else rhs_offsets
            if (arguments.Config['bab']['cut']['enabled'] and
                arguments.Config['bab']['initial_max_domains'] == 1
                and not arguments.Config['debug']['sanity_check']):
                if init_global_lb[property_idx][0] > rhs_offset:
                    print('Verified by alpha-CROWN bound!')
                    continue

            if isinstance(properties[0][0], dict):
                def _get_item(properties, key):
                    return torch.concat([
                        item[key].unsqueeze(0) for item in properties[0]], dim=0)
                x = _get_item(properties, 'X')
                data_min = _get_item(properties, 'data_min')
                data_max = _get_item(properties, 'data_max')

                for item in properties[0]:
                    assert item['eps'] == properties[0][0]['eps']
                data_dict = {
                    'eps': properties[0][0]['eps'],
                    'eps_min': properties[0][0].get('eps_min', 0),
                }
            else:
                x_range = torch.tensor(properties[0])
                data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
                data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
                x = x_range.mean(-1).reshape(vnnlib_shape)
                data_dict = None
            if 'tightened_input_bounds' in results:
                assert (
                    results['tightened_input_bounds'][0][property_idx:property_idx+1].shape
                    == data_min.shape
                )
                data_min = results['tightened_input_bounds'][0][property_idx:property_idx+1]
                data_max = results['tightened_input_bounds'][1][property_idx:property_idx+1]

            target_label_arrays = list(properties[1])
            assert len(target_label_arrays) == 1
            c, rhs, pidx = target_label_arrays[0]

            if bab_attack_enabled:
                if arguments.Config['bab']['initial_max_domains'] != 1:
                    raise ValueError(
                        'To run Bab-attack, please set initial_max_domains to 1. '
                        f'Currently it is {arguments.Config["bab"]["initial_max_domains"]}.')

                this_spec_attack_images = attack_images[:, :, property_idx].view(
                    attack_images.size(1), *attack_images.shape[3:])
            else:
                this_spec_attack_images = None

            rhs = torch.tensor(rhs, device=arguments.Config['general']['device'],
                               dtype=torch.get_default_dtype())
            if enable_incomplete and len(init_global_lb) > 1:

                ret_trim = trim_batch(
                    model_incomplete, init_global_lb, init_global_ub,
                    reference_alphas_cp, lower_bounds, upper_bounds,
                    reference_alphas, lA, property_idx, c, rhs)
                lA_trim, rhs = ret_trim['lA'], ret_trim['rhs']
                trimmed_lower_bounds = ret_trim['lower_bounds']
                trimmed_upper_bounds = ret_trim['upper_bounds']
            else:
                lA_trim = lA.copy() if lA is not None else lA
                trimmed_lower_bounds = lower_bounds
                trimmed_upper_bounds = upper_bounds

            print(f'##### Instance {index} first 10 spec matrices: ')
            print(f'{c[:10]}\nthresholds: {rhs.flatten()[:10]} ######')

            torch.cuda.empty_cache()
            gc.collect()
            c = c.to(rhs)

            if data_min.shape[0] > 1:
                l1_err_data_min = torch.norm((data_min[1:] - data_min[0:1]).view(-1), p=1)
                l1_err_data_max = torch.norm((data_max[1:] - data_max[0:1]).view(-1), p=1)
                if l1_err_data_min + l1_err_data_max < 1e-8:

                    x, data_min, data_max = x[0:1], data_min[0:1], data_max[0:1]

            time_stamp += 1
            input_split = arguments.Config['bab']['branching']['input_split']['enable']
            init_failure_idx = np.array([])
            if enable_incomplete and not input_split:
                if len(init_global_lb) > 1:

                    rlb = trimmed_lower_bounds[final_name]

                    assert len(rlb.shape) == len(rhs.shape) == 2
                    assert rlb.shape[1] == rhs.shape[1] == 1
                    init_verified_cond = rlb.flatten() > rhs.flatten()
                    init_verified_idx = torch.where(init_verified_cond)[0]
                    if len(init_verified_idx) > 0:
                        print('Initial alpha-CROWN verified for spec index '
                                f'{init_verified_idx} with bound '
                                f'{rlb[init_verified_idx].squeeze()}.')
                        l = init_global_lb[init_verified_idx].tolist()
                        bab_ret.append([index, l, 0, time.time() - start_time_bab, pidx])
                    init_failure_idx = torch.where(~init_verified_cond)[0]
                    if len(init_failure_idx) == 0:

                        continue
                    print(f'Remaining spec index {init_failure_idx} with '
                            f'bounds {rlb[init_failure_idx]} need to verify.')

                    (reference_alphas, lA_trim, x, data_min, data_max,
                    trimmed_lower_bounds, trimmed_upper_bounds, c) = prune_by_idx(
                        reference_alphas, init_verified_cond, final_name, lA_trim, x,
                        data_min, data_max, lA is not None,
                        trimmed_lower_bounds, trimmed_upper_bounds, c)

                l, nodes, ret = self.beta_crown_bab(
                    data=x, targets=init_failure_idx, time_stamp=time_stamp,
                    data_ub=data_max, data_lb=data_min, data_dict=data_dict,
                    lower_bounds=trimmed_lower_bounds, upper_bounds=trimmed_upper_bounds,
                    c=c, reference_alphas=reference_alphas, cplex_processes=cplex_processes,
                    activation_opt_params=results.get('activation_opt_params', None),
                    refined_betas=results.get('refined_betas', None), rhs=rhs[0:1],
                    reference_lA=lA_trim, attack_images=this_spec_attack_images,
                    model_incomplete=model_incomplete, timeout=timeout, vnnlib=vnnlib,
                    model=model_ori, property_idx=property_idx,
                    vnnlib_meta={
                        'property_idx': property_idx,
                        'vnnlib_id': vnnlib_id,
                        'benchmark_name': benchmark_name
                    },
                    orig_lirpa_model=orig_lirpa_model,
                )
                bab_ret.append([index, float(l), nodes,
                                time.time() - start_time_bab,
                                init_failure_idx.tolist()])
            else:
                assert arguments.Config['general']['complete_verifier'] == 'bab'
                assert not arguments.Config['bab']['attack']['enabled'], (
                    'BaB-attack must be used with incomplete verifier.')

                l, nodes, ret = self.bab(
                    data=x, targets=pidx, time_stamp=time_stamp,
                    data_ub=data_max, data_lb=data_min, c=c, data_dict=data_dict,
                    cplex_processes=cplex_processes,
                    rhs=rhs, timeout=timeout, attack_images=this_spec_attack_images,
                    vnnlib=vnnlib, model=model_ori, vnnlib_meta={
                        'property_idx': property_idx,
                        'vnnlib_id': vnnlib_id,
                        'benchmark_name': benchmark_name
                    },
                    orig_lirpa_model=orig_lirpa_model,
                )
                bab_ret.append([index, l, nodes, time.time() - start_time_bab, pidx])

            if cplex_cuts:
                solved_c_rows.append(c)
                terminate_mip_processes_by_c_matching(cplex_processes, solved_c_rows)

            timeout = timeout_threshold - (time.time() - start_time)
            if ret == 'unsafe':
                return 'unsafe-bab'
            elif ret == 'unknown' or timeout < 0:
                if arguments.Config['debug']['sanity_check'] == 'Full':
                    sanity_check_results.append(ret)
                    continue
                return 'unknown'
            elif ret != 'safe':
                raise ValueError(f'Unknown return value of bab: {ret}')
        else:
            if arguments.Config['debug']['sanity_check'] == 'Full':
                if len(sanity_check_results) == len(properties[0]):
                    print('Sanity check results:', sanity_check_results)
                    if all(result == 'unknown' for result in sanity_check_results):
                        return 'unknown'
                    else:
                        assert False, 'Sanity check failed. Something is wrong.'
            return 'safe'

    def alpha_beta(self, interm_bounds=None):
        run_mode, save_path, file_root, example_idx_list, model_ori, vnnlib_all, \
            shape, device, bab_args, debug_args, timeout_threshold, select_instance, complete_verifier = self.settings()

        for new_idx, csv_item in enumerate(example_idx_list):
            arguments.Globals['example_idx'] = new_idx
            vnnlib_id = new_idx + arguments.Config['data']['start']

            if select_instance and not vnnlib_id in select_instance:
                continue
            self.logger.record_start_time()

            print(f'\n {"%"*35} idx: {new_idx}, vnnlib ID: {vnnlib_id} {"%"*35}')
            if arguments.Config['general']['save_output']:
                arguments.Globals['out']['idx'] = new_idx

            onnx_path = None
            if run_mode != 'customized_data':
                if len(csv_item) == 3:

                    model_ori, shape, vnnlib, onnx_path = load_model_and_vnnlib(
                        file_root, csv_item)
                    arguments.Config['model']['onnx_path'] = os.path.join(file_root, csv_item[0])
                    arguments.Config['specification']['vnnlib_path'] = os.path.join(
                        file_root, csv_item[1])
                else:

                    vnnlib = read_vnnlib(os.path.join(file_root, csv_item[0]))
                    assert arguments.Config['model']['input_shape'] is not None, (
                        'vnnlib does not have shape information, '
                        'please specify by --input_shape')
                    shape = arguments.Config['model']['input_shape']
            else:
                vnnlib = vnnlib_all[new_idx]

            bab_args['timeout'] = float(bab_args['timeout'])
            timeout_threshold = bab_args['timeout']
            self.logger.update_timeout(timeout_threshold)

            model_ori.eval()
            vnnlib_shape = shape

            if isinstance(vnnlib[0][0], dict):
                x = vnnlib[0][0]['X'].reshape(vnnlib_shape)
                data_min = vnnlib[0][0]['data_min'].reshape(vnnlib_shape)
                data_max = vnnlib[0][0]['data_max'].reshape(vnnlib_shape)
            else:
                x_range = torch.tensor(vnnlib[0][0])
                data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
                data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
                x = x_range.mean(-1).reshape(vnnlib_shape)

            model_ori = model_ori.to(device)
            x, data_max, data_min = x.to(device), data_max.to(device), data_min.to(device)
            verified_status, verified_success = 'unknown', False

            model_incomplete = cplex_processes = None
            ret = {}

            orig_lirpa_model = None
            if (not verified_success
                    and self.method=='alpha' or self.method=='alpha_beta'):

                incomplete_verification_output = self.alpha_crown(
                    model_ori,
                    x,
                    data_ub=data_max,
                    data_lb=data_min,
                    vnnlib=vnnlib,
                    interm_bounds=interm_bounds
                )

                verified_status, ret, orig_lirpa_model = incomplete_verification_output

                verified_success = verified_status != 'unknown'
                model_incomplete = ret.get('model', None)

            if (not verified_success
                    and complete_verifier != 'skip'
                    and verified_status != 'unknown-mip'
                    and self.method=='beta' or self.method=='alpha_beta'):
                batched_vnnlib = batch_vnnlib(vnnlib)
                benchmark_name = (file_root.split('/')[-1]
                                  if debug_args['sanity_check'] is not None else None)
                verified_status = self.beta_crown(
                    model_ori, model_incomplete, vnnlib, batched_vnnlib, vnnlib_shape,
                    new_idx, bab_ret=self.logger.bab_ret, cplex_processes=cplex_processes,
                    timeout_threshold=timeout_threshold - (time.time() - self.logger.start_time),
                    results=ret, vnnlib_id=vnnlib_id,
                    benchmark_name=benchmark_name, orig_lirpa_model=orig_lirpa_model
                )

            del ret

            self.logger.summarize_results(verified_status, new_idx)

        self.logger.finish()
        return self.logger.verification_summary

def main(method, args_list):
    abcrown = abCrown(method=method, args=args_list)
    abcrown.alpha_beta()

if __name__== '__main__':
    main(method=sys.argv[1], args_list=sys.argv[2:])
