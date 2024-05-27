# import sys
# sys.path.append("/home/yyt/nfshare/yolov8/")
from ultralytics import YOLO

# Create a new YOLO model from scratch
# model = YOLO('/home/yyt/nfshare/yolov8/ultralytics/cfg/models/v8/yolov8.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('D:\\垃圾分类\\ultralytics-main-DCN\\modules\\DCNV2-YOLOV8.yaml')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='D:\\BaiduNetdiskDownload\\jpg_xml_14964\\newtest\\data.yaml', amp=False, epochs=300, batch=8,
                      val=True)

# Evaluate the model's performance on the validation set
# results = model.val(data='/home/yyt/nfshare/yolov8/ultralytics/cfg/datasets/hr.yaml',amp=False,epochs=2,batch=8)

success = model.export(format='onnx')