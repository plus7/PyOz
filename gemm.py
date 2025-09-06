#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from typing import NamedTuple
import struct
import math

# 学習のためにozIMMU(https://github.com/enp1s0/ozIMMU)の条件を決め打ち(入力の型はdouble、alphaとbetaは1.0と0.0決め打ち)
# してPythonに縮小移植したもの。ログ出力は除去していないし直訳臭いところもまだたくさんある。

OP_N=0
OP_T=1
MATRIX_A=0
MATRIX_B=1
SPLIT_TYPE_ORIG=0
SPLIT_TYPE_INT8=1
def get_bits_per_int8(k):
  if k == 0:
      return 0

  # Calculate ceil(log2(k))
  log2_k = 0
  while (1 << (log2_k + 1)) <= k :
      log2_k += 1

  if (1 << log2_k) != k :
      log2_k += 1

  # Return mantissa length
  return int(min(7, (31 - log2_k) / 2))

def test_get_bits_per_int8():
    print(get_bits_per_int8(1000))
    print(get_bits_per_int8(1020004))

class GemmPairConfig(NamedTuple):
    # '-1' means the original matrix
    A_id: int
    B_id: int

class SplitConfig(NamedTuple):
  # {[0] = original_type, [1] = ...}
    matrix_A_split_types: list
    matrix_B_split_types: list
    gemm_pair_config_list: list

def get_split_config(compute_mode):
    if compute_mode > 18:
        raise "invalid compute mode"

    num_split = compute_mode

    split_types = []
    split_types.append(SPLIT_TYPE_ORIG)
    for i in range(0, num_split):
        split_types.append(SPLIT_TYPE_INT8)

    #print(split_types)

    gemm_pair_list = []
    for sum_ in range(2, num_split+2):
        for j in range(1, sum_):
            if j > num_split or sum_ - j > num_split:
                continue
            gemm_pair_list.append(GemmPairConfig(A_id=j, B_id=sum_ - j))

    #print(gemm_pair_list)

    return SplitConfig(matrix_A_split_types=split_types, matrix_B_split_types=split_types, gemm_pair_config_list=gemm_pair_list)

def padded_ld(n):
    # 4: sizeof(uint32_t)
    # 1: sizeof(int8_t)
    return (int((n + 4 - 1) / 4) * 4) # nによっては形がちょっと伸びるということ

def get_slice_ld(M, op):
    m, n = M.shape
    if op == OP_N:
        return padded_ld(m)
    else:
        return padded_ld(n)

# Aの作業行列はrow majorなのでN方向が拡大する可能性あり 同様にBはM方向がそうなる
def get_slice_num_elements(M, op):
    m, n = M.shape
    if op == OP_N: # col_major
        tmp = n
    else:
        tmp = m
    return get_slice_ld(M, op) * tmp
    
double_mantissa_size = 52
double_exponent_size = 11
double_bias = 0x3ff

def reinterpret_as_uint(fp):
    return struct.unpack('Q', struct.pack('d', fp))[0]

def reinterpret_as_fp(uint):
    return struct.unpack('d', struct.pack('Q', uint))[0]

def mask_mantissa(fp):
    uint = reinterpret_as_uint(fp)
    mask = (1 << double_mantissa_size) - 1
    return uint & mask

def mask_exponent(fp):
    uint = reinterpret_as_uint(fp)
    mask = ((1 << double_exponent_size) - 1) << double_mantissa_size
    return uint & mask

def uint128_str(u):
  return str(u >> 64) + ":" + str( u & 0xFFFFFFFFFFFFFFFF )

def cut_int8(a, max_exp, num_split, mantissa_length):
  sign_flag = a > 0
  # When the input number is not normalized, don't set the implicit one bit.
  if mask_exponent(a):
      implict_one_bit = 1
  else:
      implict_one_bit = 0

  # mantissa is 128bit
  
  # 16 - 8 is sizeof(MANTISSA_T) - sizeof(INPUT_T)
  mantissa = (mask_mantissa(a) | (implict_one_bit << double_mantissa_size)) << ((16 - 8) * 8 + double_exponent_size)
  mantissa_shift_offset = (reinterpret_as_uint(max_exp) - mask_exponent(a)) >> double_mantissa_size
  shifted_mantissa = mantissa >> mantissa_shift_offset

  # B no mantissa_shift_offset is wrong
  #print("a, max_exp, mantissa, mantissa_shift_offset, shifted_mantissa", a, max_exp, uint128_str(mantissa), mantissa_shift_offset, uint128_str(shifted_mantissa))

  out = []

  for s in range(0, num_split):
      if sign_flag:
          sign = 1
      else:
          sign = -1
      int8 = (shifted_mantissa >> (16 * 8 - mantissa_length)) * sign
      shifted_mantissa = (shifted_mantissa << mantissa_length) & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF # tabaityou nanode ueni shift shita bun kesanaito dame
      out.append(int8)
  return out

def split_int8_A(A, mat_type, ldo, num_split, mantissa_length):
    m, n = A.shape
    #print("m, n, ldo, num_split, mantissa_length", m, n, ldo, num_split, mantissa_length)
    slices = []
    max_exps = []
    for i in range(0, num_split):
       slices.append(np.zeros((m, n)))
      #if mat_type == MATRIX_A:
      #  slices.append(np.zeros((m, ldo))) # m, n no docchika ldo ni suru hitsuyou ari
      #else:
      #  slices.append(np.zeros((ldo, n))) # m, n no docchika ldo ni suru hitsuyou ari
    #print(slices)

    if mat_type == MATRIX_A:
      row_size = m
      col_size = n
    else:
      row_size = n
      col_size = m

    for row_index in range(0, row_size):
        if mat_type == MATRIX_A:
          row = A[row_index]
        else:
          row = A.T[row_index]
        exps = [reinterpret_as_fp(mask_exponent(x)) for x in row]
        max_exp = 2 * max(exps)

        for i in range(0, col_size):
            out = cut_int8(row[i], max_exp, num_split, mantissa_length)
            for j in range(0, num_split):
              #print("row_index, i", row_index, i)
              if mat_type == MATRIX_A:
                slices[j][row_index, i] = out[j]
              else:
                slices[j][i, row_index] = out[j]

        max_exps.append(max_exp)
    #print(slices, max_exps)
    return (slices, max_exps)

def split_AB_int8(A, B, num_split, bits_per_int8):
    ld_int8_a = get_slice_ld(A, OP_T)
    ld_int8_b = get_slice_ld(B, OP_N)
    ret_a = split_int8_A(A, MATRIX_A, ld_int8_a, num_split, bits_per_int8)
    ret_b = split_int8_A(B, MATRIX_B, ld_int8_b, num_split, bits_per_int8)
    return (ret_a, ret_b)

def get_mantissa_loss_total(M, mat_type, mantissa_length, dist0):
    m,n = M.shape
    min_num_split = 3
    max_num_split = 18

    if dist0 == None:
        result = {} # compute_mode_t vs uint64_t
        for mode in range(3, 19):
           result[mode] = 0
    else:
        result = dist0

    if mat_type == MATRIX_A:
        row_size = m
        col_size = n
    else:
        row_size = n
        col_size = m

    for row_index in range(0, row_size):
        if mat_type == MATRIX_A:
            row = M[row_index]
        else:
            row = M.T[row_index]
        exps = [reinterpret_as_fp(mask_exponent(x)) for x in row]
        max_exp = 2 * max(exps)

        for a in row:
            if a == 0 or max_exp == 0:
                continue
            required_mantissa_space_length = ((mask_exponent(max_exp) - mask_exponent(a)) >> 52) + 53
            for num_split in range(min_num_split, max_num_split+1):
                mantissa_loss_length = 0
                mantissa_space_length = num_split * mantissa_length
                if mantissa_space_length < required_mantissa_space_length:
                    mantissa_loss_length = required_mantissa_space_length - mantissa_space_length

                result[num_split] += mantissa_loss_length
    return result

def auto_mode_select(A, B, mantissa_loss_threshold):
  m,k = A.shape
  k,n = B.shape
  bits_per_int8 = get_bits_per_int8(k)
  dist0 = get_mantissa_loss_total(A, MATRIX_A, bits_per_int8, None)
  dist = get_mantissa_loss_total(B, MATRIX_B, bits_per_int8, dist0)

  for mode in range(3, 19):
    if dist[mode] / float(m * k + k * n) <= mantissa_loss_threshold:
        return mode
  return -1 # dgemm

def accumulate_in_f64(C, C_int, mantissa_rshift):
  scale = reinterpret_as_fp( int( (double_bias - mantissa_rshift) ) << double_mantissa_size )
  m, n = C.shape
  for mi in range(0, m):
    for ni in range(0, n):
      C[mi][ni] = C[mi][ni] + float(int(C_int[mi][ni]) << 32) * scale
      #print("tid", mi,ni, "f64", C[mi][ni], "i32", int(C_int[mi][ni]), "i64", int(C_int[mi][ni]))

def axby(C, a_max_exp, b_max_exp): # alpha=1.0, beta=0.0決め打ち
  m,n = C.shape
  NewC = np.zeros((m, n))

  for mi in range(0, m):
    for ni in range(0, n):
      NewC[mi][ni] = C[mi][ni] / (1 << 44) * a_max_exp[mi] * b_max_exp[ni]
  return NewC

def gemm(A, B, compute_mode=-1):
    #compute_mode = 3 # fp64_int8_3
    if compute_mode == -1:
       compute_mode = auto_mode_select(A, B, 0.0)
    print("compute_mode = {}".format(compute_mode))
    # type check
    if not isinstance(A, np.ndarray):
        raise "type error"
    if not isinstance(B, np.ndarray):
        raise "type error"
    if A.ndim != 2 or B.ndim != 2:
        raise "dimension error"
    if A.dtype != np.float64:
        raise "element type error"
    if B.dtype != np.float64:
        raise "element type error"

    # shape check
    m_a,k_a = A.shape
    k_b,n_b = B.shape
    if k_a != k_b:
        raise "shape error"

    m = m_a
    k = k_a
    n = n_b

    C = np.zeros((m, n))

    split_config = get_split_config(compute_mode)
    num_split = len(split_config.matrix_A_split_types) - 1
    bits_per_int8 = get_bits_per_int8(k)

    a_int8_slices, b_int8_slices = split_AB_int8(A, B, num_split, bits_per_int8)
    gemm_pair_config_list = split_config.gemm_pair_config_list

    for gemm_pair_config in gemm_pair_config_list:
                                  # k no kawari
        Ai = a_int8_slices[0][gemm_pair_config.A_id-1]
        Bi = b_int8_slices[0][gemm_pair_config.B_id-1]
        C_int = Ai @ Bi
        #print(C_int)
        accumulate_in_f64(C, C_int, bits_per_int8 * (gemm_pair_config.A_id + gemm_pair_config.B_id - 2) - (7 - bits_per_int8) * 2)

    #print(C)

    a_max_exp = a_int8_slices[1]
    b_max_exp = b_int8_slices[1]
    return axby(C, a_max_exp, b_max_exp) # alpha = 1.0 beta = 0.0

if __name__ == "__main__":
    #np.random.seed(0)
    A = (np.random.randn(4, 4)).astype(np.float64)
    B = (np.random.randn(4, 4)).astype(np.float64)
    #A = np.array([[1.0, 3.0], [2.0, 4.0]]).astype(np.float64)
    #B = np.array([[5.0, 7.0], [6.0, 8.0]]).astype(np.float64)
    #A = np.array([[1.0, 4.0, 7.0, 10.0], [2.0, 5.0, 8.0, 11.0], [3.0, 6.0, 9.0, 12.0]]).astype(np.float64)
    #B = np.array([[13.0, 17.0, 21.0], [256.0, 18.0, 22.0], [15.0, 19.0, 512.0], [16.0, 20.0, 24.0]]).astype(np.float64)

    C_ozaki = gemm(A, B)
    print(C_ozaki)
    C_ref   = (A @ B)
    print(C_ref)

    abs_err = np.max(np.abs(C_ozaki - C_ref))
    rel_err = abs_err / (np.max(np.abs(C_ref)) + 1e-16)

    print("max abs error:", abs_err)
    print("max rel error:", rel_err)
