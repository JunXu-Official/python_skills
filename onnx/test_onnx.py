import onnxruntime as ort
import numpy as np

# —— 1. 创建三个推理会话 —————————————————————
sess1       = ort.InferenceSession("model1.onnx")
sess2       = ort.InferenceSession("model2.onnx")
sess_merged = ort.InferenceSession("policy_merged.onnx")

# —— 2. 打印各模型的输入/输出 名称 & 形状 —————————
print("=== Model1 ===")
print(" Inputs :", [(i.name, i.shape) for i in sess1.get_inputs()])
print(" Outputs:", [(o.name, o.shape) for o in sess1.get_outputs()], "\n")

print("=== Model2 ===")
print(" Inputs :", [(i.name, i.shape) for i in sess2.get_inputs()])
print(" Outputs:", [(o.name, o.shape) for o in sess2.get_outputs()], "\n")

print("=== Merged ===")
print(" Inputs :", [(i.name, i.shape) for i in sess_merged.get_inputs()])
print(" Outputs:", [(o.name, o.shape) for o in sess_merged.get_outputs()], "\n")

# —— 3. 构造随机测试输入 ————————————————————————
# 对 model1 和 merged，输入都是 10 维
input_name1  = sess1.get_inputs()[0].name
input_name_m = sess_merged.get_inputs()[0].name
test_input   = np.random.rand(1, 10).astype(np.float32)
print("Test input (1×10):", test_input, "\n")

# —— 4. 运行 model1 ———————————————————————————
out1 = sess1.run(None, {input_name1: test_input})[0]
print("Model1 output (1×7):", out1, "\n")

# —— 5. 构造 model2 的输入，并运行 ———————————————————
# model2 的输入名
input_name2 = sess2.get_inputs()[0].name
# 拼接 test_input[:, :5] 和 model1 的 out1 → 12 维
concat_in2 = np.concatenate([test_input[:, :5], out1], axis=1)
print("Model2 input (1×12):", concat_in2, "\n")

out2 = sess2.run(None, {input_name2: concat_in2})[0]
print("Model2 output (1×3):", out2, "\n")

# —— 6. 运行 merged.onnx ————————————————————————
out_m = sess_merged.run(None, {input_name_m: test_input})[0]
print("Merged model output (1×3):", out_m)

