import onnx
from onnx import helper, TensorProto
from onnxconverter_common import float16


# Utility: add unique prefix to all names in a graph
def prefix_graph(graph: onnx.GraphProto, prefix: str):
    for init in graph.initializer:
        init.name = prefix + init.name
    for vi in graph.input:
        vi.name = prefix + vi.name
    for vo in graph.output:
        vo.name = prefix + vo.name
    for v in graph.value_info:
        v.name = prefix + v.name
    for node in graph.node:
        if node.name:
            node.name = prefix + node.name
        node.input[:] = [prefix + x if x else x for x in node.input]
        node.output[:] = [prefix + x for x in node.output]

# 1. Load sub-models
m1 = onnx.load("model1.onnx")
m2 = onnx.load("model2.onnx")

# 2. Prefix each to avoid collisions
prefix_graph(m1.graph, "emb_")
prefix_graph(m2.graph, "mdl_")

# 3. Rename m2 input for merge
for inp in m2.graph.input:
    if inp.name == "mdl_model_input":
        inp.name = "concat31"
for node in m2.graph.node:
    node.input[:] = ["concat31" if x == "mdl_model_input" else x for x in node.input]

# 4. Define Slice node for opset9 (no 'steps' attribute)
slice_node = helper.make_node(
    op_type="Slice",
    inputs=["emb_model_input"],  # only data
    outputs=["slice15"],
    name="SliceFirst15",
    starts=[0],
    ends=[15],
    axes=[2]
)

# 5. Concat slice15 and embedding output
concat_node = helper.make_node(
    op_type="Concat",
    inputs=["slice15", "emb_model_output"],
    outputs=["concat31"],
    name="Concat15_16_to_31",
    axis=2
)

# 6. Assemble nodes and initializers
all_nodes = [slice_node] + list(m1.graph.node) + [concat_node] + list(m2.graph.node)
all_inits = list(m1.graph.initializer) + list(m2.graph.initializer)

# 7. Define merged graph I/O
new_input  = helper.make_tensor_value_info("emb_model_input", TensorProto.FLOAT, ["batch",1,18])
new_output = helper.make_tensor_value_info("mdl_model_output", TensorProto.FLOAT, ["batch",1,6])

# 8. Build and save model with opset9
merged_graph = helper.make_graph(
    all_nodes,
    "PolicyMergedGraph",
    inputs=[new_input],
    outputs=[new_output],
    initializer=all_inits
)
merged_model = helper.make_model(
    merged_graph,
    producer_name="merge_policy_models",
    ir_version=4,
    opset_imports=[helper.make_opsetid("", 9)]
)
onnx.save(merged_model, "policy_merged.onnx")
print("✅ policy_merged.onnx 已生成，采用 opset9，slice无需 steps 属性")


# 载入模型
model = onnx.load("policy_merged.onnx")

# 1. 查看 IR 版本
print("ir_version:", model.ir_version)

# 2. 查看所有 opset_imports
for opset in model.opset_import:
    print(f"opset version: {opset.version}")


#转换精度

# 1. 加载原始 FP32 模型
model_fp32 = onnx.load("policy_merged.onnx")

# 2. 转换成 FP16
model_fp16 = float16.convert_float_to_float16(
    model_fp32,
    keep_io_types=False
)

# 3. 保存 FP16 模型
onnx.save(model_fp16, "policy_merged_fp16.onnx")
print("✅ 已生成 FP16 模型：model_fp16.onnx")


