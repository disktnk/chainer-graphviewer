import numpy

import graphviewer.proto.types_pb2 as tfdtype


# Standard mappings between types_pb2.DataType values and numpy.dtypes.
_NP_TO_TF = {
    numpy.dtype('float16'): tfdtype.DT_HALF,
    numpy.dtype('float32'): tfdtype.DT_FLOAT,
    numpy.dtype('float64'): tfdtype.DT_DOUBLE,
    numpy.dtype('int32'): tfdtype.DT_INT32,
    numpy.dtype('int64'): tfdtype.DT_INT64,
    numpy.dtype('uint8'): tfdtype.DT_UINT8,
    numpy.dtype('uint16'): tfdtype.DT_UINT16,
    numpy.dtype('uint32'): tfdtype.DT_UINT32,
    numpy.dtype('uint64'): tfdtype.DT_UINT64,
    numpy.dtype('int16'): tfdtype.DT_INT16,
    numpy.dtype('int8'): tfdtype.DT_INT8,
    numpy.dtype('complex64'): tfdtype.DT_COMPLEX64,
    numpy.dtype('complex128'): tfdtype.DT_COMPLEX128,
    numpy.dtype('object'): tfdtype.DT_STRING,
    numpy.dtype('bool'): tfdtype.DT_BOOL,
    # not support qint8, quint8, qint16, quint16, qint32, bfloat16
}


def convert_dtype(dtype):
    if dtype not in _NP_TO_TF:
        raise TypeError('cannot convert the dtype')
    return _NP_TO_TF[dtype]
