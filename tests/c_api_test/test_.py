# coding: utf-8
import ctypes
import os
import sys

from platform import system

import numpy as np
from scipy import sparse


def find_lib_path():
    if os.environ.get('LIGHTGBM_BUILD_DOC', False):
        # we don't need lib_lightgbm while building docs
        return []

    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    dll_path = [curr_path,
                os.path.join(curr_path, '../../'),
                os.path.join(curr_path, '../../python-package/lightgbm/compile'),
                os.path.join(curr_path, '../../python-package/compile'),
                os.path.join(curr_path, '../../lib/')]
    if system() in ('Windows', 'Microsoft'):
        dll_path.append(os.path.join(curr_path, '../../python-package/compile/Release/'))
        dll_path.append(os.path.join(curr_path, '../../python-package/compile/windows/x64/DLL/'))
        dll_path.append(os.path.join(curr_path, '../../Release/'))
        dll_path.append(os.path.join(curr_path, '../../windows/x64/DLL/'))
        dll_path = [os.path.join(p, 'lib_lightgbm.dll') for p in dll_path]
    else:
        dll_path = [os.path.join(p, 'lib_lightgbm.so') for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if not lib_path:
        dll_path = [os.path.realpath(p) for p in dll_path]
        raise Exception('Cannot find lightgbm library file in following paths:\n' + '\n'.join(dll_path))
    return lib_path


def LoadDll():
    lib_path = find_lib_path()
    if len(lib_path) == 0:
        return None
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    return lib


LIB = LoadDll()

LIB.LGBM_GetLastError.restype = ctypes.c_char_p

dtype_float32 = 0
dtype_float64 = 1
dtype_int32 = 2
dtype_int64 = 3


def c_array(ctype, values):
    return (ctype * len(values))(*values)


def c_str(string):
    return ctypes.c_char_p(string.encode('ascii'))


def load_from_file(filename, reference):
    ref = None
    if reference is not None:
        ref = reference
    handle = ctypes.c_void_p()
    LIB.LGBM_DatasetCreateFromFile(
        c_str(filename),
        c_str('max_bin=15'),
        ref,
        ctypes.byref(handle))
    print(LIB.LGBM_GetLastError())
    num_data = ctypes.c_long()
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data))
    num_feature = ctypes.c_long()
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature))
    print('#data: %d #feature: %d' % (num_data.value, num_feature.value))
    return handle


def save_to_binary(handle, filename):
    LIB.LGBM_DatasetSaveBinary(handle, c_str(filename))


def load_from_csr(filename, reference):
    data = []
    label = []
    with open(filename, 'r') as inp:
        for line in inp.readlines():
            values = line.split('\t')
            data.append([float(x) for x in values[1:]])
            label.append(float(values[0]))
    mat = np.array(data)
    label = np.array(label, dtype=np.float32)
    csr = sparse.csr_matrix(mat)
    handle = ctypes.c_void_p()
    ref = None
    if reference is not None:
        ref = reference

    LIB.LGBM_DatasetCreateFromCSR(
        c_array(ctypes.c_int, csr.indptr),
        dtype_int32,
        c_array(ctypes.c_int, csr.indices),
        csr.data.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)),
        dtype_float64,
        ctypes.c_int64(len(csr.indptr)),
        ctypes.c_int64(len(csr.data)),
        ctypes.c_int64(csr.shape[1]),
        c_str('max_bin=15'),
        ref,
        ctypes.byref(handle))
    num_data = ctypes.c_long()
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data))
    num_feature = ctypes.c_long()
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature))
    LIB.LGBM_DatasetSetField(handle, c_str('label'), c_array(ctypes.c_float, label), len(label), 0)
    print('#data: %d #feature: %d' % (num_data.value, num_feature.value))
    return handle


def load_from_csc(filename, reference):
    data = []
    label = []
    with open(filename, 'r') as inp:
        for line in inp.readlines():
            values = line.split('\t')
            data.append([float(x) for x in values[1:]])
            label.append(float(values[0]))
    mat = np.array(data)
    label = np.array(label, dtype=np.float32)
    csr = sparse.csc_matrix(mat)
    handle = ctypes.c_void_p()
    ref = None
    if reference is not None:
        ref = reference

    LIB.LGBM_DatasetCreateFromCSC(
        c_array(ctypes.c_int, csr.indptr),
        dtype_int32,
        c_array(ctypes.c_int, csr.indices),
        csr.data.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)),
        dtype_float64,
        ctypes.c_int64(len(csr.indptr)),
        ctypes.c_int64(len(csr.data)),
        ctypes.c_int64(csr.shape[0]),
        c_str('max_bin=15'),
        ref,
        ctypes.byref(handle))
    num_data = ctypes.c_long()
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data))
    num_feature = ctypes.c_long()
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature))
    LIB.LGBM_DatasetSetField(handle, c_str('label'), c_array(ctypes.c_float, label), len(label), 0)
    print('#data: %d #feature: %d' % (num_data.value, num_feature.value))
    return handle


def load_from_mat(filename, reference):
    data = []
    label = []
    with open(filename, 'r') as inp:
        for line in inp.readlines():
            values = line.split('\t')
            data.append([float(x) for x in values[1:]])
            label.append(float(values[0]))
    mat = np.array(data)
    data = np.array(mat.reshape(mat.size), copy=False)
    label = np.array(label, dtype=np.float32)
    handle = ctypes.c_void_p()
    ref = None
    if reference is not None:
        ref = reference

    LIB.LGBM_DatasetCreateFromMat(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)),
        dtype_float64,
        mat.shape[0],
        mat.shape[1],
        1,
        c_str('max_bin=15'),
        ref,
        ctypes.byref(handle))
    num_data = ctypes.c_long()
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data))
    num_feature = ctypes.c_long()
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature))
    LIB.LGBM_DatasetSetField(handle, c_str('label'), c_array(ctypes.c_float, label), len(label), 0)
    print('#data: %d #feature: %d' % (num_data.value, num_feature.value))
    return handle


def free_dataset(handle):
    LIB.LGBM_DatasetFree(handle)


def test_dataset():
    train = load_from_file(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        '../../examples/binary_classification/binary.train'), None)
    test = load_from_mat(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../examples/binary_classification/binary.test'), train)
    free_dataset(test)
    test = load_from_csr(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../examples/binary_classification/binary.test'), train)
    free_dataset(test)
    test = load_from_csc(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../examples/binary_classification/binary.test'), train)
    free_dataset(test)
    save_to_binary(train, 'train.binary.bin')
    free_dataset(train)
    train = load_from_file('train.binary.bin', None)
    free_dataset(train)


def test_booster():
    train = load_from_mat(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '../../examples/binary_classification/binary.train'), None)
    test = load_from_mat(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../examples/binary_classification/binary.test'), train)
    booster = ctypes.c_void_p()
    LIB.LGBM_BoosterCreate(
        train,
        c_str("app=binary metric=auc num_leaves=31 verbose=0"),
        ctypes.byref(booster))
    LIB.LGBM_BoosterAddValidData(booster, test)
    is_finished = ctypes.c_int(0)
    for i in range(1, 51):
        LIB.LGBM_BoosterUpdateOneIter(booster, ctypes.byref(is_finished))
        result = np.array([0.0], dtype=np.float64)
        out_len = ctypes.c_ulong(0)
        LIB.LGBM_BoosterGetEval(
            booster,
            0,
            ctypes.byref(out_len),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if i % 10 == 0:
            print('%d iteration test AUC %f' % (i, result[0]))
    LIB.LGBM_BoosterSaveModel(booster, 0, -1, c_str('model.txt'))
    LIB.LGBM_BoosterFree(booster)
    free_dataset(train)
    free_dataset(test)
    booster2 = ctypes.c_void_p()
    num_total_model = ctypes.c_long()
    LIB.LGBM_BoosterCreateFromModelfile(
        c_str('model.txt'),
        ctypes.byref(num_total_model),
        ctypes.byref(booster2))
    data = []
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           '../../examples/binary_classification/binary.test'), 'r') as inp:
        for line in inp.readlines():
            data.append([float(x) for x in line.split('\t')[1:]])
    mat = np.array(data)
    preb = np.zeros(mat.shape[0], dtype=np.float64)
    num_preb = ctypes.c_long()
    data = np.array(mat.reshape(mat.size), copy=False)
    LIB.LGBM_BoosterPredictForMat(
        booster2,
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)),
        dtype_float64,
        mat.shape[0],
        mat.shape[1],
        1,
        1,
        25,
        c_str(''),
        ctypes.byref(num_preb),
        preb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    LIB.LGBM_BoosterPredictForFile(
        booster2,
        c_str(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           '../../examples/binary_classification/binary.test')),
        0,
        0,
        25,
        c_str(''),
        c_str('preb.txt'))
    LIB.LGBM_BoosterFree(booster2)
