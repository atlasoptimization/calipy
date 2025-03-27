#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to check broadcasting rules for CalipyTensor operations.
This includes checking the correctness of multiple operation classes:
    1. Elementwise monary CalipyTensor ops
    2. Elementwise binary CalipyTensor ops
    3. Multiplication, addition, ...  between CT and numerical classes int, float,  ..
    3. Multiplication, addition, ...  between CT and torch.Tensor class
    
    
"""

import torch
import numpy as np

from calipy.core.tensor import broadcast_dims, CalipyTensor
from calipy.core.utils import dim_assignment


# Build torch tensors
a_int = 1
a_flt = 1.0
a_np = np.ones([])

a = torch.tensor(1.0)
b = torch.ones([2])
c = torch.ones([2,1])
d = torch.ones([2,3])
e = torch.ones([2,3,4])

# Create dims
dims_full = dim_assignment(['dim_1' , 'dim_2', 'dim_3'], [2,3,4],
                           dim_descriptions = ['first_dim' , 'second_dim' , 'third_dim'])

dim_1 = dims_full[0:1]
dim_2 = dims_full[1:2]
dim_3 = dims_full[2:3]

# Invoke CalipyTensors
a_cp = CalipyTensor(a)
b_cp = CalipyTensor(b, dim_1)
c_cp = CalipyTensor(c, dim_1 + dim_2)
d_cp = CalipyTensor(d, dim_1 + dim_2)
e_cp = CalipyTensor(e, dim_1 + dim_2 + dim_3)


# build broadcasted dims
dims_aa = broadcast_dims(a_cp.bound_dims, a_cp.bound_dims)[2]
dims_ab = broadcast_dims(a_cp.bound_dims, b_cp.bound_dims)[2]
dims_ac = broadcast_dims(a_cp.bound_dims, c_cp.bound_dims)[2]
dims_ad = broadcast_dims(a_cp.bound_dims, d_cp.bound_dims)[2]
dims_ae = broadcast_dims(a_cp.bound_dims, e_cp.bound_dims)[2]

dims_ba = broadcast_dims(b_cp.bound_dims, a_cp.bound_dims)[2]
dims_bb = broadcast_dims(b_cp.bound_dims, b_cp.bound_dims)[2]
dims_bc = broadcast_dims(b_cp.bound_dims, c_cp.bound_dims)[2]
dims_bd = broadcast_dims(b_cp.bound_dims, d_cp.bound_dims)[2]
dims_be = broadcast_dims(b_cp.bound_dims, e_cp.bound_dims)[2]

dims_ca = broadcast_dims(c_cp.bound_dims, a_cp.bound_dims)[2]
dims_cb = broadcast_dims(c_cp.bound_dims, b_cp.bound_dims)[2]
dims_cc = broadcast_dims(c_cp.bound_dims, c_cp.bound_dims)[2]
dims_cd = broadcast_dims(c_cp.bound_dims, d_cp.bound_dims)[2]
dims_ce = broadcast_dims(c_cp.bound_dims, e_cp.bound_dims)[2]

dims_da = broadcast_dims(d_cp.bound_dims, a_cp.bound_dims)[2]
dims_db = broadcast_dims(d_cp.bound_dims, b_cp.bound_dims)[2]
dims_dc = broadcast_dims(d_cp.bound_dims, c_cp.bound_dims)[2]
dims_dd = broadcast_dims(d_cp.bound_dims, d_cp.bound_dims)[2]
dims_de = broadcast_dims(d_cp.bound_dims, e_cp.bound_dims)[2]

dims_ea = broadcast_dims(e_cp.bound_dims, a_cp.bound_dims)[2]
dims_eb = broadcast_dims(e_cp.bound_dims, b_cp.bound_dims)[2]
dims_ec = broadcast_dims(e_cp.bound_dims, c_cp.bound_dims)[2]
dims_ed = broadcast_dims(e_cp.bound_dims, d_cp.bound_dims)[2]
dims_ee = broadcast_dims(e_cp.bound_dims, e_cp.bound_dims)[2]

# Check broadcasted dims
assert(dims_aa.sizes == [])
assert(dims_ab.sizes == [2])
assert(dims_ac.sizes == [2,1])
assert(dims_ad.sizes == [2,3])
assert(dims_ae.sizes == [2,3,4])

assert(dims_ba.sizes == [2])
assert(dims_bb.sizes == [2])
assert(dims_bc.sizes == [2,1])
assert(dims_bd.sizes == [2,3])
assert(dims_be.sizes == [2,3,4])

assert(dims_ca.sizes == [2,1])
assert(dims_cb.sizes == [2,1])
assert(dims_cc.sizes == [2,1])
assert(dims_cd.sizes == [2,3])
assert(dims_ce.sizes == [2,3,4])

assert(dims_da.sizes == [2,3])
assert(dims_db.sizes == [2,3])
assert(dims_dc.sizes == [2,3])
assert(dims_dd.sizes == [2,3])
assert(dims_de.sizes == [2,3,4])

assert(dims_ea.sizes == [2,3,4])
assert(dims_eb.sizes == [2,3,4])
assert(dims_ec.sizes == [2,3,4])
assert(dims_ed.sizes == [2,3,4])
assert(dims_ee.sizes == [2,3,4])


# Special cases

# Interleaved dims : Finding dim, supersequence then extending to it
dims_p = dim_assignment(['dim_2', 'dim_3'], [3,4])
dims_q = dim_assignment(['dim_1', 'dim_2', 'dim_4'], [2,3,5])
dims_pq = broadcast_dims(dims_p, dims_q)

assert(dims_pq[2].sizes == [2,3,4,5]) # Should deliver result dim of shape [2,3,4,5]
assert(dims_pq[0].sizes == [1,3,4,1]) # Should deliver expanded dim_1 of shape [1,3,4,1]
assert(dims_pq[1].sizes == [2,3,1,5]) # Should deliver expanded dim_2 of shape [2,3,1,5]

# Check correctness of elementwise operations on CalipyTensors

# Addition +

# Build sums
add_aa = a_cp + a_cp
add_ab = a_cp + b_cp
add_ac = a_cp + c_cp
add_ad = a_cp + d_cp
add_ae = a_cp + e_cp

add_ba = b_cp + a_cp
add_bb = b_cp + b_cp
add_bc = b_cp + c_cp
add_bd = b_cp + d_cp
add_be = b_cp + e_cp

add_ca = c_cp + a_cp
add_cb = c_cp + b_cp
add_cc = c_cp + c_cp
add_cd = c_cp + d_cp
add_ce = c_cp + e_cp

add_da = d_cp + a_cp
add_db = d_cp + b_cp
add_dc = d_cp + c_cp
add_dd = d_cp + d_cp
add_de = d_cp + e_cp

add_ea = e_cp + a_cp
add_eb = e_cp + b_cp
add_ec = e_cp + c_cp
add_ed = e_cp + d_cp
add_ee = e_cp + e_cp

# Check dims of sums
assert(add_aa.bound_dims.sizes == dims_aa.sizes)
assert(add_ab.bound_dims.sizes == dims_ab.sizes)
assert(add_ac.bound_dims.sizes == dims_ac.sizes)
assert(add_ad.bound_dims.sizes == dims_ad.sizes)
assert(add_ae.bound_dims.sizes == dims_ae.sizes)

assert(add_ba.bound_dims.sizes == dims_ba.sizes)
assert(add_bb.bound_dims.sizes == dims_bb.sizes)
assert(add_bc.bound_dims.sizes == dims_bc.sizes)
assert(add_bd.bound_dims.sizes == dims_bd.sizes)
assert(add_be.bound_dims.sizes == dims_be.sizes)

assert(add_ca.bound_dims.sizes == dims_ca.sizes)
assert(add_cb.bound_dims.sizes == dims_cb.sizes)
assert(add_cc.bound_dims.sizes == dims_cc.sizes)
assert(add_cd.bound_dims.sizes == dims_cd.sizes)
assert(add_ce.bound_dims.sizes == dims_ce.sizes)

assert(add_da.bound_dims.sizes == dims_da.sizes)
assert(add_db.bound_dims.sizes == dims_db.sizes)
assert(add_dc.bound_dims.sizes == dims_dc.sizes)
assert(add_dd.bound_dims.sizes == dims_dd.sizes)
assert(add_de.bound_dims.sizes == dims_de.sizes)

assert(add_ea.bound_dims.sizes == dims_ea.sizes)
assert(add_eb.bound_dims.sizes == dims_eb.sizes)
assert(add_ec.bound_dims.sizes == dims_ec.sizes)
assert(add_ed.bound_dims.sizes == dims_ed.sizes)
assert(add_ee.bound_dims.sizes == dims_ee.sizes)


# Multiplication *

# Division /



