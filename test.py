#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pyoz


A = (np.random.randn(4, 4)).astype(np.float64)
B = (np.random.randn(4, 4)).astype(np.float64)

C_ref   = (A @ B)
print("reference result")
print(C_ref)
print("--------------------------------------")

modes = [-1]
modes.extend(range(3, 18))

for mode in modes:
    C_ozaki = pyoz.gemm(A, B, mode, True)
    print(C_ozaki)

    abs_err = np.max(np.abs(C_ozaki - C_ref))
    rel_err = abs_err / (np.max(np.abs(C_ref)) + 1e-16)

    print("max abs error:", abs_err)
    print("max rel error:", rel_err)
    print("--------------------------------------")