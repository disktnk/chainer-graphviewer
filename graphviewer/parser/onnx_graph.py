from graphviewer.proto.attr_value_pb2 import AttrValue
from graphviewer.proto.graph_pb2 import GraphDef
from graphviewer.proto.node_def_pb2 import NodeDef
from graphviewer.proto.tensor_shape_pb2 import TensorShapeProto
from graphviewer.proto.versions_pb2 import VersionDef


# This script is from
# https://github.com/lanpa/tensorboardX/blob/4e7bb739cb1a70191c58411d8a01536a82c9b4bd/tensorboardX/onnx_graph.py  # NOQA


def get_graphdef_from_file(path):
    import onnx
    model = onnx.load(path)
    return parse(model.graph)


def parse(graph):
    nodes_proto, nodes = [], []
    import itertools
    for node in itertools.chain(graph.input, graph.output):
        nodes_proto.append(node)

    for node in nodes_proto:
        shapreproto = TensorShapeProto(
            dim=[TensorShapeProto.Dim(size=d.dim_value) for d in node.type.tensor_type.shape.dim])
        nodes.append(NodeDef(
            name=node.name.encode(encoding='utf_8'),
            op='Variable',
            input=[],
            attr={
                'dtype': AttrValue(type=node.type.tensor_type.elem_type),
                'shape': AttrValue(shape=shapreproto),
            }
        ))

    for node in graph.node:
        attr = []
        for s in node.attribute:
            attr.append(' = '.join([str(f[1]) for f in s.ListFields()]))
        attr = ', '.join(attr).encode(encoding='utf_8')
        nodes.append(NodeDef(
            name=node.output[0].encode(encoding='utf_8'),
            op=node.op_type,
            input=node.input,
            attr={'parameters': AttrValue(s=attr)},
        ))
    mapping = {}
    for node in nodes:
        mapping[node.name] = node.op + '_' + node.name

    return GraphDef(node=nodes, versions=VersionDef(producer=22))
