import mediapipe as mp
from tqdm import tqdm
import numpy as np
import cv2
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,        # 是静态图片还是连续视频帧
                       max_num_hands=2,                # 最多检测几只手
                       min_detection_confidence=0.8,   # 置信度阈值
                       min_tracking_confidence=0.5)    # 追踪阈值
mpDraw = mp.solutions.drawing_utils 


def process_one_hand(mp_hand_result,hand_idx):

    # 给定结果簇与索引，进行单只手的处理，输出单手骨架向量
    hand_21 = mp_hand_result.multi_hand_world_landmarks[hand_idx]           # 获取该手的21个关键点坐标

    # 以食指指根[5]为Z轴
    vec_z = np.array([(hand_21.landmark[0].z - hand_21.landmark[5].z),      # 深度估计化为x向
                    ((hand_21.landmark[5].x - hand_21.landmark[0].x)),      # 图像x向化为y向
                    ((hand_21.landmark[0].y - hand_21.landmark[5].y))])     # 图像y向化为z向

    norm_vec_z = np.linalg.norm(vec_z)
    hand_size_scaler = 10.0 / norm_vec_z
    normal_vec_z = vec_z * hand_size_scaler
    new_e_z = normal_vec_z / 10.0                                           # 新的基底ez

    vec_y_z = np.array([(hand_21.landmark[0].z - hand_21.landmark[17].z),   # 深度估计化为x向
        ((hand_21.landmark[17].x - hand_21.landmark[0].x) ),                # 图像x向化为y向
        ((hand_21.landmark[0].y - hand_21.landmark[17].y) )])               # 图像y向化为z向

    normal_vec_y_z =  vec_y_z * hand_size_scaler
    horizontal_on_z = np.dot(normal_vec_y_z,new_e_z) * new_e_z
    vec_y =  horizontal_on_z - normal_vec_y_z

    new_e_y = vec_y / np.linalg.norm(vec_y)                                 # 新的基底ey
    new_e_x = np.cross(new_e_y, new_e_z)                                    # 新的基底ex

    new_e_mat = np.array([new_e_x,new_e_y,new_e_z])                         # 旋转矩阵

    for hand_point_idx in range(len(hand_21.landmark)):
        if hand_point_idx == 0:
            hand_point_vec = np.array([-hand_21.landmark[hand_point_idx].z,hand_21.landmark[hand_point_idx].x,-hand_21.landmark[hand_point_idx].y])
        elif hand_point_idx == 5:
            continue
        else:
            hand_point_vec = np.row_stack((hand_point_vec,[[-hand_21.landmark[hand_point_idx].z,hand_21.landmark[hand_point_idx].x,-hand_21.landmark[hand_point_idx].y]]))
    
    hand_point_vec += [hand_21.landmark[0].z, -hand_21.landmark[0].x, hand_21.landmark[0].y]

    normal_hand_point_vec = hand_point_vec * hand_size_scaler
    normal_hand_point_vec = np.matmul(new_e_mat,normal_hand_point_vec.T).T

    normal_hand_point_vec[0] = [hand_21.landmark[0].z,hand_21.landmark[0].x,hand_21.landmark[0].y]  # 第一项用绝对坐标
    normal_hand_point_vec = np.row_stack((normal_hand_point_vec,new_e_mat))

    return normal_hand_point_vec



def process_frame_vec(img):

    img = cv2.flip(img, 1)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = hands.process(img_RGB)
    img.flags.writeable = True
    one_hand_vec_len = 23
    end_idx = one_hand_vec_len * 2 - 1
    vec_out = np.zeros((one_hand_vec_len * 2, 3))   # 总输出

    if results.multi_hand_world_landmarks:

        if len(results.multi_hand_world_landmarks) == 1:
            
            #一只手的时候处理
            one_hand_vec = process_one_hand(results,0)
            
            for circle_idx in range(one_hand_vec_len):
                vec_out[circle_idx] = one_hand_vec[circle_idx]
                vec_out[end_idx-circle_idx] = one_hand_vec[circle_idx]
        
        elif len(results.multi_hand_world_landmarks) == 2:
            
            #两只手的时候处理
            first_hand_vec = process_one_hand(results,0)
            second_hand_vec = process_one_hand(results,1)
            for circle_idx in range(one_hand_vec_len):
                vec_out[circle_idx] = first_hand_vec[circle_idx]
                vec_out[end_idx-circle_idx] = second_hand_vec[circle_idx]

    return vec_out, results
    # else:
    #     return False
            

def draw_hands(frame, hand_result):
    # 如果有识别到，就去绘制
    frame = cv2.flip(frame, 1)
    if hand_result.multi_hand_world_landmarks:
        for hand_idx in range(len(hand_result.multi_hand_landmarks)):
            hand_21 = hand_result.multi_hand_landmarks[hand_idx]
            mpDraw.draw_landmarks(frame, hand_21, mp_hands.HAND_CONNECTIONS)

    return frame


# cap = cv2.VideoCapture(0)
# cap.open(0)
# # cap.set(cv2.CAP_PROP_FPS,10)
# np.set_printoptions(suppress=True) # 不以科学计数法显示

# while cap.isOpened():
#     start_time = time.time()
#     success, frame = cap.read()
#     if not success:
#         break
#     ori_frame = frame.copy() # 原始图像保留一份
#     hands_vector, hand_result = process_frame_vec(frame)
#     frame = draw_hands(frame, hand_result)
#     end_time = time.time()
#     FPS = 1/(end_time - start_time)
#     FPS_str = "{}".format(FPS)
#     cv2.putText(frame, FPS_str, (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 1, cv2.LINE_AA)
#     cv2.imshow('my_window', frame)
#     if cv2.waitKey(1) in [ord('q'),27]: # 按键盘上的q或esc退出（在英文输入法下）
#         break
# cap.release()
# cv2.destroyAllWindows()


import os

DATA_PATH = os.path.join('手势数据集') 
record_class = '好'
record_class_en = 'good'
record_start_idx = 0

cap = cv2.VideoCapture(0)
cap.open(0)
np.set_printoptions(suppress=True) # 不以科学计数法显示

while cap.isOpened():

    success, frame = cap.read()
    if not success:
        break
    
    ori_frame = frame.copy()        # 原始图像保留一份
    hands_vector, hand_result = process_frame_vec(frame)
    frame = draw_hands(frame, hand_result)
    tips_str = 'Now you can record the idx {} data in \'{}\' class'.format(record_start_idx,record_class_en)
    key = cv2.waitKey(1)
    cv2.putText(frame, tips_str, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255, 0), 1, cv2.LINE_AA)
    cv2.imshow('my_window', frame)

    if key in [ord('q'),27]: # 按键盘上的q或esc退出（在英文输入法下）
        break
    elif key in [ord(' ')]:
        record_start_idx += 1

    
cap.release()
cv2.destroyAllWindows()
