import cv2
import mediapipe as mp
import numpy as np
import os, time
# colab에서 학습시킨 모델 불러오기
from tensorflow.keras.models import load_model

model = load_model('LSTM_Model2.h5')
seq_length = 30 # LSTM을 사용하고자 seq 길이 설정(window)

gestures = ['rock', 'scissors', 'paper']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
# angle 데이터 저장(x,y,z,visbility), 30
seq = []

# 영상 저장 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while cap.isOpened():
    success, img = cap.read()

    if not success:
        break
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            # joint 초기화
            joint = np.zeros((21,4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:3] # Child joint

            v = v2- v1 # [20,3], 팔목(0)과 각 손가락 관절 사이의 벡터
            
            # Normalize(유닛 vector(=크기가 1) => 유클리디안 distance) 
            # 이유>>? 벡터의 방향만 나타내기 위함
            # 벡터 / 벡터의 길이(스칼라)
            v = v / np.expand_dims(np.linalg.norm(v, axis=1), axis=1) # output (20,) 축 하나 추가(np.expand_dims) for 계산

            # 두개의 단위벡터 내적으로 Theta 구하기
            A = v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:]
            B = v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]

            angle = np.arccos(np.einsum('ij,ij->i', A, B))
            angle = np.degrees(angle)

            data = np.concatenate([joint.flatten(), angle])

            # seq angle data 저장
            seq.append(data)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            # seq데이터 float64 ==> float32와 차원하나 추가(expand_dims)
            seq_array= np.array(seq[-30:], dtype=np.float32)

            seq_data = np.expand_dims(seq_array, axis=0)

            # .squeeze ==> 차원 중 사이즈가 1인 것을 찾아 스칼라값으로 바꿔 해당 차원을 제거
            y_pred = model.predict(seq_data)

            # action 0 1 2 
            i_pred = int(np.argmax(y_pred))
            
            # 인식 신뢰도(.squeeze 사용시 )
            # conf = y_pred[i_pred]

            # if conf < 0.9:
            #     continue
            
            # action data 저장
            action = gestures[i_pred]

            # image.shape (480, 640, 3)
            target_org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20))
            # org == 손목 0 부분에 맞춤
            cv2.putText(img, text= action, org=target_org, fontFace= cv2.FONT_HERSHEY_SIMPLEX,
                     fontScale=1, color=(255,255,0), thickness=2)
    # 영상저장
    out.write(img)
    cv2.imshow('result',img)
    if cv2.waitKey(1) == ord('q'):
        break