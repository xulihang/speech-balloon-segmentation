"""Convert bubble-pop's float16 ONNX (float16 input/output) to a float32 IO model.

The Java binding uses float32 tensors (onnxruntime-java 1.20 has no simple
float16 tensor constructor), so the Java project loads models/model_fp32.onnx.
Only the precision changes: every float16 initializer and the IO types become
float32, so the output differs from the fp16 model only by rounding.

Usage: python java/tools/convert_to_fp32.py <model_fp16.onnx> <model_fp32.onnx>
"""
import sys

import onnx
from onnx import TensorProto, numpy_helper


def convert(src, dst):
    model = onnx.load(src)
    graph = model.graph
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype("float32")
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT16:
                    arr = numpy_helper.to_array(attr.t).astype("float32")
                    attr.t.CopyFrom(numpy_helper.from_array(arr, "value"))
        elif node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
    for vi in list(graph.input) + list(graph.output):
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print("saved", dst)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
