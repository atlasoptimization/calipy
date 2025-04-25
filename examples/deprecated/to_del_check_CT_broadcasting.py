#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to check broadcasting rules for CalipyTensor operations.
This includes checking the correctness of multiple operation classes:
    1. Elementwise monary CalipyTensor ops
    2. Elementwise binary CalipyTensor ops
    3. Multiplication, addition, ...  between CT and numerical classes int, float,  ..
    3. Multiplication, addition, ...  between CT and torch.Tensor class
    
There are two broadcasting mechanisms: If an elementwise operation is performed
on a CalipyTensor and a regular torch.tensor, pytorch broadcasting rules are
applied and the new result will have dims from the CalipyTensor and possibly some
autodims. If an elementwise operation is applied to two CalipyTensors, the shortest
possible supersequence of dims is constructed that contain dims_1 and dims_2 as
subsequences; then dims_1 and dims_2 are extended to match that supersequence and 
bdroadcasted via torch. In short, this second, dimension aware approach, aligns
dims by name before using standard broadcasting.
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


# build broadcasted dims of CalipyTensors
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
# Different from pytorch broadcasting since matching by dims means extension of
# [dim1] by [dim1, dim2] to [dim1, dim2] with sizes [2,1]:
#   [dim1, dim2] : [2,1] by [dim1] [2] -> [dim1, dim2] [2,1]
assert(dims_bb.sizes == [2])
assert(dims_bc.sizes == [2,1])
assert(dims_bd.sizes == [2,3])
assert(dims_be.sizes == [2,3,4])

assert(dims_ca.sizes == [2,1])
assert(dims_cb.sizes == [2,1]) # Different: [dim1, dim2] : [2,1] by [dim1] [2] -> [dim1, dim2] [2,1]
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

# Build sums of two CalipyTensors
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



# Multiplication * of CalipyTensors

mult_aa = a_cp * a_cp
mult_ab = a_cp * b_cp
mult_ac = a_cp * c_cp
mult_ad = a_cp * d_cp
mult_ae = a_cp * e_cp

mult_ba = b_cp * a_cp
mult_bb = b_cp * b_cp
mult_bc = b_cp * c_cp
mult_bd = b_cp * d_cp
mult_be = b_cp * e_cp

mult_ca = c_cp * a_cp
mult_cb = c_cp * b_cp
mult_cc = c_cp * c_cp
mult_cd = c_cp * d_cp
mult_ce = c_cp * e_cp

mult_da = d_cp * a_cp
mult_db = d_cp * b_cp
mult_dc = d_cp * c_cp
mult_dd = d_cp * d_cp
mult_de = d_cp * e_cp

mult_ea = e_cp * a_cp
mult_eb = e_cp * b_cp
mult_ec = e_cp * c_cp
mult_ed = e_cp * d_cp
mult_ee = e_cp * e_cp

# Check dims of products
assert(mult_aa.bound_dims.sizes == dims_aa.sizes)
assert(mult_ab.bound_dims.sizes == dims_ab.sizes)
assert(mult_ac.bound_dims.sizes == dims_ac.sizes)
assert(mult_ad.bound_dims.sizes == dims_ad.sizes)
assert(mult_ae.bound_dims.sizes == dims_ae.sizes)

assert(mult_ba.bound_dims.sizes == dims_ba.sizes)
assert(mult_bb.bound_dims.sizes == dims_bb.sizes)
assert(mult_bc.bound_dims.sizes == dims_bc.sizes)
assert(mult_bd.bound_dims.sizes == dims_bd.sizes)
assert(mult_be.bound_dims.sizes == dims_be.sizes)

assert(mult_ca.bound_dims.sizes == dims_ca.sizes)
assert(mult_cb.bound_dims.sizes == dims_cb.sizes)
assert(mult_cc.bound_dims.sizes == dims_cc.sizes)
assert(mult_cd.bound_dims.sizes == dims_cd.sizes)
assert(mult_ce.bound_dims.sizes == dims_ce.sizes)

assert(mult_da.bound_dims.sizes == dims_da.sizes)
assert(mult_db.bound_dims.sizes == dims_db.sizes)
assert(mult_dc.bound_dims.sizes == dims_dc.sizes)
assert(mult_dd.bound_dims.sizes == dims_dd.sizes)
assert(mult_de.bound_dims.sizes == dims_de.sizes)

assert(mult_ea.bound_dims.sizes == dims_ea.sizes)
assert(mult_eb.bound_dims.sizes == dims_eb.sizes)
assert(mult_ec.bound_dims.sizes == dims_ec.sizes)
assert(mult_ed.bound_dims.sizes == dims_ed.sizes)
assert(mult_ee.bound_dims.sizes == dims_ee.sizes)



# Division / of CalipyTensors

div_aa = a_cp / a_cp
div_ab = a_cp / b_cp
div_ac = a_cp / c_cp
div_ad = a_cp / d_cp
div_ae = a_cp / e_cp

div_ba = b_cp / a_cp
div_bb = b_cp / b_cp
div_bc = b_cp / c_cp
div_bd = b_cp / d_cp
div_be = b_cp / e_cp

div_ca = c_cp / a_cp
div_cb = c_cp / b_cp
div_cc = c_cp / c_cp
div_cd = c_cp / d_cp
div_ce = c_cp / e_cp

div_da = d_cp / a_cp
div_db = d_cp / b_cp
div_dc = d_cp / c_cp
div_dd = d_cp / d_cp
div_de = d_cp / e_cp

div_ea = e_cp / a_cp
div_eb = e_cp / b_cp
div_ec = e_cp / c_cp
div_ed = e_cp / d_cp
div_ee = e_cp / e_cp

# Check dims of products
assert(div_aa.bound_dims.sizes == dims_aa.sizes)
assert(div_ab.bound_dims.sizes == dims_ab.sizes)
assert(div_ac.bound_dims.sizes == dims_ac.sizes)
assert(div_ad.bound_dims.sizes == dims_ad.sizes)
assert(div_ae.bound_dims.sizes == dims_ae.sizes)

assert(div_ba.bound_dims.sizes == dims_ba.sizes)
assert(div_bb.bound_dims.sizes == dims_bb.sizes)
assert(div_bc.bound_dims.sizes == dims_bc.sizes)
assert(div_bd.bound_dims.sizes == dims_bd.sizes)
assert(div_be.bound_dims.sizes == dims_be.sizes)

assert(div_ca.bound_dims.sizes == dims_ca.sizes)
assert(div_cb.bound_dims.sizes == dims_cb.sizes)
assert(div_cc.bound_dims.sizes == dims_cc.sizes)
assert(div_cd.bound_dims.sizes == dims_cd.sizes)
assert(div_ce.bound_dims.sizes == dims_ce.sizes)

assert(div_da.bound_dims.sizes == dims_da.sizes)
assert(div_db.bound_dims.sizes == dims_db.sizes)
assert(div_dc.bound_dims.sizes == dims_dc.sizes)
assert(div_dd.bound_dims.sizes == dims_dd.sizes)
assert(div_de.bound_dims.sizes == dims_de.sizes)

assert(div_ea.bound_dims.sizes == dims_ea.sizes)
assert(div_eb.bound_dims.sizes == dims_eb.sizes)
assert(div_ec.bound_dims.sizes == dims_ec.sizes)
assert(div_ed.bound_dims.sizes == dims_ed.sizes)
assert(div_ee.bound_dims.sizes == dims_ee.sizes)



# Now do the same for tensors with generic dimensions or standard torch tensors
# Under the hood, torch.tensors a are wrapped in a_cp = CalipyTensor(a) which 
# produces a CalipyTensor with generic dims, so the expressions a_cp + b_cp and
# CalipyTensor(a) + b_cp are equal.
# The results are equivalent to what standard pytorch produces during broadcasting

# Invoke CalipyTensors
a_gcp = CalipyTensor(a)
b_gcp = CalipyTensor(b)
c_gcp = CalipyTensor(c)
d_gcp = CalipyTensor(d)
e_gcp = CalipyTensor(e)


# build broadcasted dims of CalipyTensors
dims_aag = broadcast_dims(a_cp.bound_dims, a_gcp.bound_dims)[2]
dims_abg = broadcast_dims(a_cp.bound_dims, b_gcp.bound_dims)[2]
dims_acg = broadcast_dims(a_cp.bound_dims, c_gcp.bound_dims)[2]
dims_adg = broadcast_dims(a_cp.bound_dims, d_gcp.bound_dims)[2]
dims_aeg = broadcast_dims(a_cp.bound_dims, e_gcp.bound_dims)[2]

dims_bag = broadcast_dims(b_cp.bound_dims, a_gcp.bound_dims)[2]
dims_bbg = broadcast_dims(b_cp.bound_dims, b_gcp.bound_dims)[2]
dims_bcg = broadcast_dims(b_cp.bound_dims, c_gcp.bound_dims)[2]
# dims_bdg = broadcast_dims(b_cp.bound_dims, d_gcp.bound_dims)[2] # Not broadcastable: [2], [2,3]
# dims_beg = broadcast_dims(b_cp.bound_dims, e_gcp.bound_dims)[2] # Not broadcastable: [2], [2,3,4]

dims_cag = broadcast_dims(c_cp.bound_dims, a_gcp.bound_dims)[2]
dims_cbg = broadcast_dims(c_cp.bound_dims, b_gcp.bound_dims)[2]
dims_ccg = broadcast_dims(c_cp.bound_dims, c_gcp.bound_dims)[2]
dims_cdg = broadcast_dims(c_cp.bound_dims, d_gcp.bound_dims)[2]
# dims_ceg = broadcast_dims(c_cp.bound_dims, e_gcp.bound_dims)[2] # Not broadcastable: [2,1], [2,3,4]

dims_dag = broadcast_dims(d_cp.bound_dims, a_gcp.bound_dims)[2]
# dims_dbg = broadcast_dims(d_cp.bound_dims, b_gcp.bound_dims)[2] # Not broadcastable: [2,3], [2]
dims_dcg = broadcast_dims(d_cp.bound_dims, c_gcp.bound_dims)[2]
dims_ddg = broadcast_dims(d_cp.bound_dims, d_gcp.bound_dims)[2]
# dims_deg = broadcast_dims(d_cp.bound_dims, e_gcp.bound_dims)[2] # Not broadcastable: [2,3], [2,3,4]

dims_eag = broadcast_dims(e_cp.bound_dims, a_gcp.bound_dims)[2]
# dims_ebg = broadcast_dims(e_cp.bound_dims, b_gcp.bound_dims)[2] # Not broadcastable: [2,3,4], [2]
# dims_ecg = broadcast_dims(e_cp.bound_dims, c_gcp.bound_dims)[2] # Not broadcastable: [2,3,4], [2,1]
# dims_edg = broadcast_dims(e_cp.bound_dims, d_gcp.bound_dims)[2] # Not broadcastable: [2,3,4], [2,3]
dims_eeg = broadcast_dims(e_cp.bound_dims, e_gcp.bound_dims)[2]


# Check broadcasted dims
assert(dims_aag.sizes == [])
assert(dims_abg.sizes == [2])
assert(dims_acg.sizes == [2,1])
assert(dims_adg.sizes == [2,3])
assert(dims_aeg.sizes == [2,3,4])

assert(dims_bag.sizes == [2])
assert(dims_bbg.sizes == [2])

# Different from cp broadcasting since matching from right means extension of
# [2,1] by [2] to [2,2]
assert(dims_bcg.sizes == [2,2]) 

assert(dims_cag.sizes == [2,1])
assert(dims_cbg.sizes == [2,2]) # Different: [2,1] by [2] -> [2,2]
assert(dims_ccg.sizes == [2,1])
assert(dims_cdg.sizes == [2,3])

assert(dims_dag.sizes == [2,3])
assert(dims_dcg.sizes == [2,3])
assert(dims_ddg.sizes == [2,3])

assert(dims_eag.sizes == [2,3,4])
assert(dims_eeg.sizes == [2,3,4])


# Build sums of CalipyTensors with torch.tensors
add_aat = a_cp + a
add_abt = a_cp + b
add_act = a_cp + c
add_adt = a_cp + d
add_aet = a_cp + e

add_bat = b_cp + a
add_bbt = b_cp + b
add_bct = b_cp + c
# add_bdt = b_cp + d # Not broadcastable: [2], [2,3]
# add_bet = b_cp + e # Not broadcastable: [2], [2,3,4]

add_cat = c_cp + a
add_cbt = c_cp + b
add_cct = c_cp + c
add_cdt = c_cp + d
# add_cet = c_cp + e # Not broadcastable: [2,1], [2,3,4]

add_dat = d_cp + a
# add_dbt = d_cp + b # Not broadcastable: [2,3], [2]
add_dct = d_cp + c
add_ddt = d_cp + d
# add_det = d_cp + e # Not broadcastable: [2,3], [2,3,4]

add_eat = e_cp + a
# add_ebt = e_cp + b # Not broadcastable: [2,3,4], [2]
# add_ect = e_cp + c # Not broadcastable: [2,3,4], [2,1]
# add_edt = e_cp + d # Not broadcastable: [2,3,4], [2,3]
add_eet = e_cp + e

# Check dims of sums
assert(add_aat.bound_dims.sizes == dims_aag.sizes)
assert(add_abt.bound_dims.sizes == dims_abg.sizes)
assert(add_act.bound_dims.sizes == dims_acg.sizes)
assert(add_adt.bound_dims.sizes == dims_adg.sizes)
assert(add_aet.bound_dims.sizes == dims_aeg.sizes)

assert(add_bat.bound_dims.sizes == dims_bag.sizes)
assert(add_bbt.bound_dims.sizes == dims_bbg.sizes)
assert(add_bct.bound_dims.sizes == dims_bcg.sizes)

assert(add_cat.bound_dims.sizes == dims_cag.sizes)
assert(add_cbt.bound_dims.sizes == dims_cbg.sizes)
assert(add_cct.bound_dims.sizes == dims_ccg.sizes)
assert(add_cdt.bound_dims.sizes == dims_cdg.sizes)

assert(add_dat.bound_dims.sizes == dims_dag.sizes)
assert(add_dct.bound_dims.sizes == dims_dcg.sizes)
assert(add_ddt.bound_dims.sizes == dims_ddg.sizes)

assert(add_eat.bound_dims.sizes == dims_eag.sizes)
assert(add_eet.bound_dims.sizes == dims_eeg.sizes)

# Left addition and right addition are equal since by definition in CalipyTensor.__add__
# we have __add__(self, other) = __radd__(self, other)

assert(((a_cp + e_cp).tensor == (e_cp + a_cp).tensor).all())

# Addition, Multplication, Division also works naturally with Python integer and floats

# Addition
2 + b_cp
b_cp + 2
2.0 + b_cp
b_cp + 2.0
torch.tensor(np.ones([2])) + b_cp
b_cp + torch.tensor(np.ones([2]))

# Multiplication
2 * b_cp
b_cp * 2
2.0 * b_cp
b_cp * 2.0
torch.tensor(np.ones([2])) * b_cp
b_cp * torch.tensor(np.ones([2]))

# Division
2 / b_cp
b_cp / 2
2.0 / b_cp
b_cp / 2.0
torch.tensor(2*np.ones([2])) / b_cp
b_cp / torch.tensor(2*np.ones([2]))


# Numpy arrays behave different because they have their own __add__ methods.
b_cp + np.ones([2]) # This works partially
# np.ones([2]) + b_cp # This errors out since addition not defined in np
b_cp * np.ones([2]) # This works partially
# np.ones([2]) * b_cp # This errors out since addition not defined in np
b_cp / np.ones([2]) # This works partially
# np.ones([2]) / b_cp # This errors out since addition not defined in np

# If the tensors cannot be broadcasted together, an exception is raised:
# d_cp + CalipyTensor(torch.ones([20,20]) , dim_assignment(['dim_1', 'dim_2'])) # Shape mismatch
# d_cp + CalipyTensor(torch.ones([20,20]) , dim_assignment(['dim_2', 'dim_1'])) # Dim order mismatch

# Similar things hold also for multiplication and division.