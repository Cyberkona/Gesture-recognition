import mediapipe as mp
from tqdm import tqdm
import numpy as np
import cv2
import time
import yaml

# 导入手部识别模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,        # 是静态图片还是连续视频帧
                       max_num_hands=2,                # 最多检测几只手
                       min_detection_confidence=0.8,   # 置信度阈值
                       min_tracking_confidence=0.5)    # 追踪阈值
mpDraw = mp.solutions.drawing_utils 