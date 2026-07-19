# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The main script for the back-propagation algorithm
"""
import torch
import data
from data import random_data, process_pool, routing_region
import util
import model
import timeit
import argparse
from torch_scatter import scatter_max
import tracemalloc
import os
import numpy as np
import sys
import math

# ----------------------------------------------------------------------------
# 1. Setup Output Logging
# ----------------------------------------------------------------------------
old_out = sys.stdout
class StAmpedOut:
    """Stamped stdout."""
    nl = True
    start = timeit.default_timer()
    def write(self, x):
        if x == '\n':
            old_out.write(x)
            self.nl = True
        elif self.nl:
            old_out.write('[%.3f] %s' % (float(timeit.default_timer() - self.start), x))
            self.nl = False
        else:
            old_out.write(x)
    def flush(self):          
        pass
    
sys.stdout = StAmpedOut()
tracemalloc.start()

# ----------------------------------------------------------------------------
# 2. Argument Parsing
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--xmax', type=int, default=50)
parser.add_argument('--ymax', type=int, default=50)
parser.add_argument('--capacity', type=float, default=1.0)
parser.add_argument('--net_num', type=int, default=50)
parser.add_argument('--max_pin_num', type=int, default=3)
parser.add_argument('--net_size', type=int, default=10)
parser.add_argument('--data_path', type=str, default='/scratch/weili3/cu-gr-2/run/bsg.pt')

# --- TUNED DEFAULTS (LR=0.8, Iter=2000) ---
parser.add_argument('--lr', type=float, default=0.8)
parser.add_argument('--optimizer', type=str, default='rmsprop')
parser.add_argument('--scheduler', type=str, default='constant')
parser.add_argument('--t', type=float, default=1, help = "temperature scale")
parser.add_argument('--tree_t', type=float, default=0.9)
parser.add_argument('--iter', type=int, default=2000)
parser.add_argument('--epoch_iter', type=int, default=1)
parser.add_argument('--act', type=str, default='sigmoid')
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--via_coeff', type=float, default=4)
parser.add_argument('--wl_coeff', type=float, default=0.5)
parser.add_argument('--overflow_coeff', type=float, default=1)
parser.add_argument('--celu_alpha', type=float, default=2.0)
parser.add_argument('--act_scale', type=float, default=0.5)
parser.add_argument('--via_layer', type=float, default=1.5)
parser.add_argument('--use_gumble', type=bool, default=True)
parser.add_argument('--pattern_level', type=int, default=1)
parser.add_argument('--z_step', type=int, default=3)
parser.add_argument('--max_z', type=int, default=10)
parser.add_argument('--c_step', type=int, default=3)
parser.add_argument('--max_c', type=int, default=20)
parser.add_argument('--max_c_out_ratio', type=float, default=5)
parser.add_argument('--pin_ratio', type=float, default=1)
parser.add_argument('--local_net_ratio', type=float, default=1)
parser.add_argument('--add_CZ', type=bool, default=False)
parser.add_argument('--add_via', type=bool, default=True)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--select_threshold', type=float, default=1.0)
parser.add_argument('--read_new_tree', type=bool, default=False)
parser.add_argument('--tree_file_path', type=str, default=None)
parser.add_argument('--output_name', type=str, default='output')
parser.add_argument('--use_ilp_metric', type=bool, default=False)

# NEW ARGUMENTS
parser.add_argument('--warmstart_file', type=str, default='warmstart_test5.npz', help="Path to GNN output .npz")
parser.add_argument('--save_target', type=str, default=None, help="Save final probs as .npz for GNN training")

args = parser.parse_args()
args.device = 'cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu'
args.data_name = args.data_path.split('/')[-1].split('.')[0]
args.lr = args.lr / args.epoch_iter

os.makedirs('./tmp', exist_ok=True)
os.makedirs('./CUGR2_guide', exist_ok=True) 

start = timeit.default_timer()

# ----------------------------------------------------------------------------
# 3. Data Loading
# ----------------------------------------------------------------------------
assert args.data_path is not None, "Random data is not used now"

results = torch.load(args.data_path, map_location='cpu')
RouteNets = results['net']
RoutingRegion = results['region']
args.xmax = RoutingRegion.xmax
args.ymax = RoutingRegion.ymax
args.net_num = len(RouteNets)
RoutingRegion3D = results["region3D"]
num_layer = len(RoutingRegion3D.cap_mat_3D[0]) + len(RoutingRegion3D.cap_mat_3D[1])
args.num_layer = num_layer
args.hor_first = RoutingRegion3D.hor_first
args.num_hor_layer = len(RoutingRegion3D.cap_mat_3D[0])
args.num_ver_layer = len(RoutingRegion3D.cap_mat_3D[1])
args.via_layer = float(np.sqrt(num_layer)) * args.via_layer

# Rebuild Capacity Matrix
RoutingRegion.cap_mat = [torch.stack(RoutingRegion3D.cap_mat_3D[0]).sum(0),torch.stack(RoutingRegion3D.cap_mat_3D[1]).sum(0)]
RoutingRegion.cap_mat = [RoutingRegion.cap_mat[0] * args.capacity, RoutingRegion.cap_mat[1] * args.capacity]
results['edge_length'] = [results['edge_length'][1], results['edge_length'][0]] 

# Calculate Demands
hor_edge_demand, ver_edge_demand, hor_pin_demand, ver_pin_demand = util.get_pin_demand(RouteNets, args, results['layers'], results['edge_length'])
hor_local_net_demand, ver_local_net_demand = util.get_local_net(RouteNets, args)

RoutingRegion.cap_mat[0] = RoutingRegion.cap_mat[0] - hor_local_net_demand * args.local_net_ratio
RoutingRegion.cap_mat[1] = RoutingRegion.cap_mat[1] - ver_local_net_demand * args.local_net_ratio

if args.use_ilp_metric is False:
    RoutingRegion.cap_mat[0] = RoutingRegion.cap_mat[0] - torch.tensor(hor_edge_demand) * args.pin_ratio 
    RoutingRegion.cap_mat[1] = RoutingRegion.cap_mat[1] - torch.tensor(ver_edge_demand) * args.pin_ratio

hor_edge_length = torch.tensor(results['edge_length'][0]).to(args.device).reshape(1,-1)
ver_edge_length = torch.tensor(results['edge_length'][1]).to(args.device).reshape(-1,1)
RoutingRegion.to(args.device)
hor_pin_demand = torch.tensor(hor_pin_demand).to(args.device)
ver_pin_demand = torch.tensor(ver_pin_demand).to(args.device)
m2_pitch = results['m2_pitch']
min_unit_length_short_cost = min([layer['unit_length_short_cost'] for layer in results['layers']])

util.print_data_stat(args)
print("Data generation time: ", timeit.default_timer() - start)

# ----------------------------------------------------------------------------
# 4. Candidate Generation / Loading
# ----------------------------------------------------------------------------
start = timeit.default_timer()
if os.path.exists('./tmp/' + args.data_name + '_candidate_pool.pt') and args.read_new_tree is False:
    candidate_pool = torch.load('./tmp/' + args.data_name + '_candidate_pool.pt')
    print("candidate pool loaded")
else:
    candidate_pool = util.get_initial_candidate_pool(RouteNets, args.xmax, args.ymax, device=args.device, edge_length=results['edge_length'], pattern_level=args.pattern_level, max_z=args.max_z, z_step=args.z_step, c_step=args.c_step, max_c=args.max_c, max_c_out_ratio=args.max_c_out_ratio)
    torch.save(candidate_pool, './tmp/' + args.data_name + '_candidate_pool.pt')

# --- FIX: pool_generation_time definition ---
pool_generation_time = timeit.default_timer() - start
print("Initial candidate pool generation time: ", pool_generation_time)

if os.path.exists('./tmp/' + args.data_name + '_p_index.pt') and args.read_new_tree is False:
    p_index, p_index_full, p_index2pattern_index, hor_path, ver_path, wire_length_count, via_info, tree_p_index, tree_index_per_candidate, tree_p_index2pattern_index, tree_p_index_full = torch.load('./tmp/' + args.data_name + '_p_index.pt', map_location=args.device)
else:
    p_index, p_index_full, p_index2pattern_index, hor_path, ver_path, wire_length_count, via_info, tree_p_index, tree_index_per_candidate, tree_p_index2pattern_index, tree_p_index_full = process_pool(candidate_pool, args.xmax, args.ymax, device=args.device)
    torch.save((p_index, p_index_full, p_index2pattern_index, hor_path, ver_path, wire_length_count, via_info, tree_p_index, tree_index_per_candidate, tree_p_index2pattern_index, tree_p_index_full), './tmp/' + args.data_name + '_p_index.pt')

if args.read_new_tree is False: 
    tree_p_index = None

# ----------------------------------------------------------------------------
# 5. Model Initialization & WARM START (NORMALIZED)
# ----------------------------------------------------------------------------
net = model.Net(p_index, pattern_level=args.pattern_level, device=args.device, use_gumble=args.use_gumble, tree_p_index=tree_p_index, tree_index_per_candidate=tree_index_per_candidate).to(args.device)

if os.path.exists(args.warmstart_file):
    print(f"[DeepDGR] Loading warmstart from {args.warmstart_file}...")
    try:
        warmstart = np.load(args.warmstart_file)
        logits = None
        
        if 'logits' in warmstart:
             logits = torch.tensor(warmstart['logits']).float().to(args.device)
             # Per-subnet centering: preserve relative preferences within each subnet
             # but prevent global drift
             for s_idx in range(len(p_index) - 1):
                 s_start = p_index[s_idx]
                 s_end = p_index[s_idx + 1]
                 if s_end > s_start:
                     subnet_logits = logits[s_start:s_end]
                     logits[s_start:s_end] = subnet_logits - subnet_logits.mean()
             logits = logits.clamp(-5.0, 5.0)  # prevent explosion
        elif 'probabilities' in warmstart:
             probs = torch.tensor(warmstart['probabilities']).float().to(args.device)
             logits = torch.log(probs.clamp(min=1e-6))
             # Per-subnet centering
             for s_idx in range(len(p_index) - 1):
                 s_start = p_index[s_idx]
                 s_end = p_index[s_idx + 1]
                 if s_end > s_start:
                     subnet_logits = logits[s_start:s_end]
                     logits[s_start:s_end] = subnet_logits - subnet_logits.mean()
             logits = logits.clamp(-5.0, 5.0)

        if logits is not None:
            if logits.shape[0] == net.p.shape[0]:
                net.p = torch.nn.Parameter(logits, requires_grad=True)
                print(f"[DeepDGR] Warmstart loaded (per-subnet centered, clamped [-5,5])!")
                print(f"[DeepDGR]   logit stats: min={logits.min():.3f} max={logits.max():.3f} "
                      f"mean={logits.mean():.3f} std={logits.std():.3f}")
            else:
                print(f"[DeepDGR] WARNING: Shape mismatch. Graph has {net.p.shape[0]} candidates, but warmstart has {logits.shape[0]}. Skipping warmstart.")
    except Exception as e:
        print(f"[DeepDGR] Failed to load warmstart: {e}")
else:
    print(f"[DeepDGR] Warmstart file {args.warmstart_file} not found. Starting from scratch.")

config = { "lr": args.lr, "t": args.t, "tree_t": args.tree_t }

if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(net.parameters(), weight_decay=args.weight_decay, betas=(args.beta1, 0.999), lr=config["lr"])
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(net.parameters(), weight_decay=args.weight_decay, lr=config["lr"])
elif args.optimizer == 'rmsprop':
    optimizer = torch.optim.RMSprop(net.parameters(), weight_decay=args.weight_decay, lr=config["lr"])
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.weight_decay, lr=config["lr"])

if args.scheduler == 'constant':
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.iter/10), gamma=0.8)

# ----------------------------------------------------------------------------
# 6. Training Loop
# ----------------------------------------------------------------------------
overflow_cost_list = []
via_cost_list = []
current_best_cost = 1e10
current_best_continue_cost = 1e10
best_cost = 1e10
worst_cost = 0
cost_list = []

temperature = 1
tree_temperature = 1

hor_cap = RoutingRegion.cap_mat[0].flatten()
ver_cap = RoutingRegion.cap_mat[1].flatten()
hor_batch = math.ceil(hor_cap.numel() / args.epoch_iter)
ver_batch = math.ceil(ver_cap.numel() / args.epoch_iter)

print("Start training...")

start = timeit.default_timer()
p_generation_time = 0
cost_inference_time = 0
back_time = 0
ilp_time = 0
ilp_obj = 1e10
lp_obj = 1e10

for i in range(args.iter):
    update_iteration = int(args.iter/10)
    if i % update_iteration == 0 and i != 0:
        temperature = temperature * config["t"]
        tree_temperature = tree_temperature * config["tree_t"]  
        if 'overflow_cost' in locals():
            print("iter %d, overflow cost: %.3f; via cost: %.3f; wl cost: %.3f; max_overflow: %.1f" % 
                  (i, overflow_cost.cpu().item()*args.epoch_iter, via_cost.cpu().item(), wire_length_cost.cpu().item(), max_overflow.cpu().item()))

    hor_shuffled_indices = torch.randperm(hor_cap.numel())
    ver_shuffled_indices = torch.randperm(ver_cap.numel())

    for j in range(args.epoch_iter):
        p_start = timeit.default_timer()
        p, candidate_p, tree_p = net.forward(temperature, tree_temperature)
        p_generation_time += timeit.default_timer() - p_start
        
        cost_start = timeit.default_timer()
        overflow_cost, via_cost, wire_length_cost, max_overflow, hor_overflow, ver_overflow = model.objective_function(
            RoutingRegion, hor_path, ver_path, wire_length_count, via_info, p, args,
            hor_pin_demand, ver_pin_demand, hor_edge_length, ver_edge_length, min_unit_length_short_cost, m2_pitch,
            iteration=j, hor_batch_size=hor_batch, ver_batch_size=ver_batch, 
            hor_shuffled_indices=hor_shuffled_indices, ver_shuffled_indices=ver_shuffled_indices
        )
        cost_inference_time += timeit.default_timer() - cost_start
        
        back_start = timeit.default_timer()
        total_cost = overflow_cost*args.overflow_coeff*args.epoch_iter + wire_length_cost*args.wl_coeff + via_cost*args.via_coeff
        
        optimizer.zero_grad()
        total_cost.backward()
        optimizer.step()
        back_time += timeit.default_timer() - back_start
    
    scheduler.step()

    cost_list.append(float(total_cost.detach().cpu()))
    overflow_cost_list.append(float(overflow_cost.detach().cpu()))
    via_cost_list.append(float(via_cost.detach().cpu()))

    if args.add_CZ:
        if i == (4*update_iteration) or i == (8*update_iteration):
            cz_level = 2
            CZ_result = util.add_CZ(net.p, p_index_full, p_index2pattern_index, hor_path, ver_path, wire_length_count, via_info, tree_index_per_candidate, 
                                    candidate_pool, hor_overflow, ver_overflow, cz_level, args, results['edge_length'])
            if CZ_result is not None:
                p_index_full, p_index2pattern_index, hor_path, ver_path, wire_length_count, via_count, via_map, tree_index_per_candidate, new_p, candidate_pool = CZ_result
                print("iter %d: %d/%d new candidates are generated "% (i, new_p.shape[0] - net.p.shape[0], net.p.shape[0]))
                net.p = torch.nn.Parameter(new_p.float(), requires_grad = True)
                net.p_full_index = p_index_full
                if net.tree_p_index is not None:
                    net.tree_index_per_candidate = tree_index_per_candidate
                
                if args.optimizer == 'adam':
                    optimizer = torch.optim.Adam(net.parameters(), weight_decay=args.weight_decay, betas=(args.beta1, 0.999), lr=config["lr"])
                elif args.optimizer == 'rmsprop':
                    optimizer = torch.optim.RMSprop(net.parameters(), weight_decay=args.weight_decay, lr=config["lr"])
                elif args.optimizer == 'sgd':
                    optimizer = torch.optim.SGD(net.parameters(), weight_decay=args.weight_decay, lr=config["lr"])
                
                via_info = (via_map, via_count)

    if cost_list[-1] < current_best_continue_cost:
        current_best_continue_cost = cost_list[-1]
    
    if i == args.iter - 1:
        current_best_cost = (overflow_cost + wire_length_cost*args.wl_coeff + via_cost*args.via_coeff).cpu().item()

train_time = (timeit.default_timer() - start)
best_cost = current_best_cost
best_continue_cost = current_best_continue_cost
worst_cost = current_best_cost

print("best cost: ", best_cost, "overflow cost: ", float(overflow_cost.cpu().detach()), "via cost: ", via_cost_list[-1])

# ----------------------------------------------------------------------------
# 7. Saving Results
# ----------------------------------------------------------------------------

# --- SAVE TARGET FOR FEEDBACK LOOP ---
if args.save_target:
    with torch.no_grad():
        # FIX: Positional args ONLY to avoid TypeError
        final_p, _, _ = net.forward(1.0, 1.0)
    np.savez(args.save_target, probabilities=final_p.cpu().numpy())
    print(f"[DeepDGR] ? Saved feedback targets to {args.save_target}")
# -------------------------------------

if args.read_new_tree is True:
    selected_tree_full_index = util.write_tree_result(RouteNets, tree_p, tree_p_index_full, tree_p_index2pattern_index, args.data_name, write_tree=args.read_new_tree)
    selected_p_full_index = selected_tree_full_index[tree_index_per_candidate]
    p_index2pattern_index[(selected_p_full_index == False).cpu()] = 0

util.write_CUGR_input(RouteNets, candidate_p, p_index_full, candidate_pool, p_index2pattern_index, args.data_name + '_' + args.output_name, args.select_threshold)

# Stats CSV
import csv
if not os.path.exists('./step1.csv'):
    with open('./step1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        row_list = []
        for arg in vars(args):
            row_list.append(arg)
        row_list += ['num_candidates','pool_generation_time', 'train_time', 'ilp_time','ilp_obj', 'best_cost','overflow_cost', 'via_cost', 'wire_length_cost','max_overflow', 'worst_cost', 'lp_obj', 'best_continue_cost', 'p_generation_time', 'cost_inference_time', 'back_time', 'peak_gpu_memory (MB)','peak_cpu_memory (MB)']
        writer.writerow(row_list)

torch.cuda.empty_cache()
with open('./step1.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    row_list = []
    for arg in vars(args):
        row_list.append(getattr(args, arg))
    # FIX: pool_generation_time is now guaranteed to be defined
    row_list += [p_index[-1], pool_generation_time, train_time, ilp_time, ilp_obj, best_cost, overflow_cost.cpu().item(), via_cost.cpu().item(), wire_length_cost.cpu().item(), max_overflow.cpu().item(), worst_cost, lp_obj, best_continue_cost, p_generation_time, cost_inference_time, back_time, torch.cuda.max_memory_allocated()/(1024*1024), tracemalloc.get_traced_memory()[1]/(1024*1024)]
    writer.writerow(row_list)