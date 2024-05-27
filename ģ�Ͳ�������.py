import torch
from torchinfo import summary
import time


def load_model(model_path):
    # 加载模型，设置为评估模式
    model = torch.load(model_path)
    model.eval()
    return model


def calculate_parameters_and_flops(model, input_size=(1, 3, 640, 640)):
    # 使用torchinfo计算模型的参数量和FLOPs
    model_summary = summary(model, input_size=input_size, verbose=0)
    total_params = model_summary.total_params
    total_flops = model_summary.total_mult_adds  # 注意：在不同版本的torchinfo中，FLOPs可能有不同的命名
    return total_params, total_flops


def measure_inference_time(model, input_tensor):
    # 测量单次推理时间
    start_time = time.time()
    with torch.no_grad():
        _ = model(input_tensor)
    end_time = time.time()
    inference_time = end_time - start_time
    return inference_time


model_paths = {
    'yolov3': 'd:/Users/admin/Desktop/论文数据/yolov3-tiny/weights/bestC3.pt',
    'yolov5': 'd:/Users/admin/Desktop/论文数据/yolov5s/weights/bestC3.pt',
    'yolov8': 'd:/Users/admin/Desktop/论文数据/v8官方模型的数据/weights/bestC3.pt',
    'yolo自己的': 'd:/Users/admin/Desktop/论文数据/自己的模型的数据/weights/bestC3.pt',
}

input_tensor = torch.randn(1, 3, 640, 640)

for model_name, model_path in model_paths.items():
    model = load_model(model_path)
    total_params, total_flops = calculate_parameters_and_flops(model)
    inference_time = measure_inference_time(model, input_tensor)

    print(f"模型: {model_name}")
    print(f"总参数量: {total_params}")
    print(f"总FLOPs: {total_flops}")
    print(f"推理时间: {inference_time} 秒")
    print("----------")
