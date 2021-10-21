import cv2
import mediapipe as mp
import numpy as np
import os 
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd 

gestures = {0: 'rock', 1 : 'scissor', 2 : 'paper'}

# colab에서 학습한 모델 load
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

# mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1, # 한개의 손만 인식
    min_detection_confidence=0.5,
    # 1.0에 가까울수록 확실한 확률이 높은 손가락만 감지

    min_tracking_confidence=0.5 # 둘다 default 0.5
    # 1.0에 가까울수록 추적율이 높은 것만 감지
)
# 영상 저장 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_ml.avi', fourcc, 20.0, (640,480))

while cap.isOpened():
    success, img = cap.read()

    if not success:
        break
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            # print(hand_landmark) x, y, z
            hand_landmark = res.landmark

            # print(coord) x, y, z, visibility 좌표( ) 84개
            coord = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand_landmark])

            # + angle 15개
            v1 = coord[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:3] # Parent coord
            v2 = coord[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:3] # Child coord

            v = v2- v1 # [20,3]
    
            # Normalize(유닛 vector => 유클리디안 distance), v == (20,3)
            # 벡터 / 벡터의 길이(스칼라)
            v = v / np.expand_dims(np.linalg.norm(v, axis=1), axis=1) # (20,1) 축 하나 추가 for 계산

            A = v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:]
            B = v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]

            # Radian 값(= unit vector를 내적한 값의 arccos를 구하면 관절 사이의 각도 구할 수 있다.)
            angle = np.arccos(np.einsum('nt,nt->n', A, B)) # 각 행과의 곱 ==> 1D로 변환 / [15,]

            # degrees 변경(=gesture_train), float64 ==> float32와 차원하나 추가(expand_dims)
            angle = np.degrees(angle)

            # 총 99개의 데이터
            coord_row = coord.flatten()
            row = np.concatenate([coord_row, angle])
            
            # train model 적용
            X = pd.DataFrame([row])
            pred = model.predict(X)[0] # 예측 target
            pred_proba = model.predict_proba(X)[0] # 예측 확률 [[0, 1, 2]] 확률


            idx = int(pred)
            if idx in gestures.keys():
                gesture_name = gestures[idx]

                # gesture_name 손목에 위치
                # image.shape (480, 640, 3)
                target_org = (int(res.landmark[0].x*img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20))

                # gesture_name 이미지에 적용
                cv2.putText(img, text=gesture_name, org=target_org, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1, 
                        color= (255,255,0), thickness=2)
            
                # 가장 높은 확률을 출력
                proba_text = str(pred_proba[np.argmax(pred_proba)])
                cv2.putText(img, text=proba_text, org=(10,30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255,255,255), thickness=1)

            # 이미지에 landmark 그리기
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    # 영상저장
    out.write(img)
    cv2.imshow('result',img)
    
    if cv2.waitKey(1) == ord('q'):
        break


