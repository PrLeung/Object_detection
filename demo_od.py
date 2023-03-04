from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
import torch
import time
import streamlit as st
import torchvision
import numpy as np

# 定义app的标题
st.title("Object Detection Demo")
st.write("")

model_selector = st.selectbox(
    'Please select a model for detecting：',
    ('YOLO', 'Fast R-CNN')
)
# 图像上传控件定义
file_up = st.file_uploader("Upload an image", type="jpg")

# 定义对象检测模型的类别标签；
inst_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
inst_class_to_idx = {cls: idx for (idx, cls) in enumerate(inst_classes)}


# 定义推理函数
def predict_yolo(image):
    """Return predictions.

    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: none
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # 对输入图像进行预处理，模型推理，打印时间
    img = Image.open(image)
    img = np.asarray(img)

    time_start = time.time()
    model.eval()
    results = model([img], size=640)  # batch of images
    time_end = time.time()
    time_sum = time_end - time_start

    data = []
    # Get results
    preds = results.pandas().xyxy[0]
    for _, row in preds.iterrows():
        x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
        label = row["name"]
        data.append({'pos': [x1, y1, x2, y2], 'label': label})

    st.write('Just', time_sum, 'second!')

    time_start = time.time()
    output_labels = [item['label'] for item in data]
    output_boxes = [item['pos'] for item in data]
    output_boxes = torch.tensor(output_boxes)

    transform = transforms.Compose([
        transforms.ToTensor()])
    images = transform(img) * 255.0
    images = images.byte()

    result = draw_bounding_boxes(images, boxes=output_boxes, labels=output_labels, width=5)
    st.image(result.permute(1, 2, 0).numpy(), caption='Processed Image.', use_column_width=True)
    time_end = time.time()
    time_sum = time_end - time_start
    st.write('Draw', time_sum, 'second!')
    return preds


# 定义推理函数
def predict_rcnn(image):
    """Return predictions.

    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: none
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # 定义输入图像的预处理方法
    transform = transforms.Compose([
        transforms.ToTensor()])

    # 对输入图像进行预处理，模型推理，打印时间
    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    time_start = time.time()
    model.eval()
    outputs = model(batch_t)

    time_end = time.time()
    time_sum = time_end - time_start
    st.write('Just', time_sum, 'second!')

    time_start = time.time()
    # 将预测分数大于指定阈值的对象画框标记在图像上。
    score_threshold = .8
    output_labels = [inst_classes[label] for label in outputs[0]['labels'][outputs[0]['scores'] > score_threshold]]
    output_boxes = outputs[0]['boxes'][outputs[0]['scores'] > score_threshold]
    images = transform(img) * 255.0
    images = images.byte()
    result = draw_bounding_boxes(images, boxes=output_boxes, labels=output_labels, width=5)
    st.image(result.permute(1, 2, 0).numpy(), caption='Processed Image.', use_column_width=True)
    time_end = time.time()
    time_sum = time_end - time_start
    st.write('Draw', time_sum, 'second!')
    return outputs


# 主函数
if file_up is not None and model_selector is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    if model_selector == "Fast R-CNN":
        labels = predict_rcnn(file_up)
    elif model_selector == "YOLO":
        labels = predict_yolo(file_up)
    st.write('Is the detected target accurate in your opinion?')
    col1,col2=st.columns(2)
    with col1:
        if st.button('Yes'):
            st.write('Thanks for your use!')
    with col2:
        if st.button('No'):
            st.write('Sorry, please select another model and try again!')
