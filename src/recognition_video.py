import os
import cv2
import numpy as np
import argparse
from model import CNN3  # 确保你有 CNN3 模型文件
from utils import index2emotion, cv2_img_add_text  # 确保 utils 文件中包含这两个函数
from blazeface import blaze_detect  # 确保 BlazeFace 检测模块可用


def load_model():
    """
    加载本地模型
    :return: 训练好的模型
    """
    model = CNN3()
    model.load_weights('./models/fer2013/cnn3_best.weights.h5')  # 替换为你训练好的权重路径
    return model


def generate_faces(face_img, img_size=48):
    """
    将探测到的人脸进行增广
    :param face_img: 灰度化的单个人脸图
    :param img_size: 目标图片大小
    :return: 增广后的图片数组
    """
    face_img = face_img / 255.0
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img)
    resized_images.append(face_img[2:45, :])
    resized_images.append(face_img[1:47, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images


def predict_expression(video_path, output_path):
    """
    从视频中检测表情并保存结果到新视频
    :param video_path: 输入视频路径
    :param output_path: 输出视频路径
    :return: None
    """
    # 加载模型
    model = load_model()

    border_color = (0, 0, 0)  # 黑框
    font_color = (255, 255, 255)  # 白字

    # 打开视频文件
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 获取视频参数
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    while True:
        ret, frame = capture.read()
        if not ret:
            print("视频播放结束或出错。")
            break

        # 灰度化并检测人脸
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = blaze_detect(frame)  # 使用 BlazeFace 检测人脸

        # 如果检测到人脸
        if faces is not None and len(faces) > 0:
            for (x, y, w, h) in faces:
                face = frame_gray[y: y + h, x: x + w]
                faces_imgs = generate_faces(face)
                results = model.predict(faces_imgs)
                result_sum = np.sum(results, axis=0).reshape(-1)
                label_index = np.argmax(result_sum, axis=0)
                emotion = index2emotion(label_index)

                # 在视频帧上绘制结果
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), border_color, thickness=2)
                frame = cv2_img_add_text(frame, emotion, x + 30, y + 30, font_color, 20)

        # 保存当前帧到输出视频
        out.write(frame)

        # 实时显示当前帧（可选）
        cv2.imshow("Expression Recognition (press esc to exit)", frame)

        # 按 ESC 退出
        if cv2.waitKey(30) & 0xFF == 27:
            break

    # 释放资源
    capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 使用 argparse 获取命令行参数
    parser = argparse.ArgumentParser(description="视频表情识别并保存结果到新视频")
    parser.add_argument('--video_path', type=str, default='./temp/test.mp4', help='输入视频路径')
    parser.add_argument('--output_path', type=str, default='./output/result.mp4', help='输出视频路径')

    # 解析命令行参数
    args = parser.parse_args()

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    # 调用预测和保存函数
    predict_expression(args.video_path, args.output_path)
