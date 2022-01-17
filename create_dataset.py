import cv2
import mediapipe as mp
import numpy as np
import os, time


# 저장하고자 하는 gestures
gestures = ['rock', 'scissors', 'paper']
seq_length = 30 # LSTM을 사용하고자 seq 길이 설정(window)
action_recoding = 30 # 30초 동안 기록

# .npy data 저장시 이름에 시간 포함
# 현재 폴더에 new_dataset 폴더 생성
created_time = int(time.time())
os.makedirs('new_dataset', exist_ok=True)

# Mediapipe solution API 사용(Hands)
mp_hands = mp.solutions.hands # 손의 관절 위치를 인식할 수 있는 모델
mp_drawing = mp.solutions.drawing_utils # 관절(landmark 그리기 위함)
hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence=0.5,
    # 1.0에 가까울수록 확실한 확률이 높은 손가락만 감지
    min_tracking_confidence=0.5
    # 1.0에 가까울수록 추적율이 높은 것만 감지
)
# 영상 저장 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('data.avi', fourcc, 20.0, (640,480))

# webcam 사용
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # gesture 마다 저장
    for idx, action in enumerate(gestures):
        # action data 저장
        data = []

        success, img = cap.read() # img은 이미지를 numpy array로 변경

        # success == True or False로 반환
        if not success:
            break

        img = cv2.flip(img, 1) # 이미지 반전

        cv2.putText(img, text= f'recoding {action}', org= (10,30), fontFace= cv2.FONT_HERSHEY_SIMPLEX,
                     fontScale=1, color=(255,255,255), thickness=1)
        
        cv2.imshow('img', img)

         # waitKey() 의 리턴값은 키보드로 입력한 키와 동일한 아스키 값
        cv2.waitKey(3000) # ==> 3초
        # 1000단위로 설정하면 1초 대기(action 취하기전 준비)

        start_time = time.time()
        # 30초 동안 반복
        while time.time() - start_time <= action_recoding: # 경과시간 <= 30초
            success, img = cap.read()

            img = cv2.flip(img,1)

            # Mediapipe는 RGB color를 사용하기 때문에 변경 시키고
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # hands 적용(관절 위치 탐지)
            result = hands.process(img)

            # 원상태로 복원
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 

            if result.multi_hand_landmarks is not None:
                # 손이 여러개 들어왔을 경우  
                # max_num_hands = 1 지금은 하나로 설정
                for res in result.multi_hand_landmarks:
                    # res = 각각의 손
                    joint = np.zeros((21,4)) # 손 관절을 저장할 joint(21개의 landmark)
                    for j, lm in enumerate(res.landmark): # 21개의 landmark 
                        # joint 값 저장
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        # lm = 관절의 포인트가 저장됨 (x, y, z, vis), 총 21개의 좌표

                    # pixel로 표현 1280*720
                    # mediapipe의 경우는 상대좌표로 나타냄(0-1사이 값이나옴)

                    # x, y = [0.0, 1.0]각각 이미지 너비와 높이로 정규화된 랜드마크 좌표
                    # z = 손목의 깊이를 원점으로 하여 랜드마크 깊이를 나타내며, 랜드마크가 카메라에 가까울수록 값이 작음
                    # visibility = [0.0, 1.0]이미지에서 랜드마크가 보일 가능성(존재하고 가려지지 않은)을 나타내는 값
                    

                    # 관절사이 각도 계산(= new_dataset값)    
                    # landmark 사이 길이(관절의 vector를 구해줌), # visibility 제외 #
                    # 0 1 2 3 4
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

                    # arccos return == Radian
                    # einsum == dot product( 내적 ), 각 행과의 곱 ==> 1D로 변환
                    angle = np.arccos(np.einsum('ij,ij->i', A, B)) 
                        # [15,] = 15개의 데이터(각 landmark와의 각도)

                    # degrees 변경(=gesture_train), float64 ==> float32
                    # radian * (180*pi)
                    angle = np.degrees(angle)
                    angle_label = np.array([angle], dtype=np.float32)

                    # idx = action index
                    angle_label = np.append(angle_label, idx)

                    # action_label data
                    # joint.flatten() = 100개의 데이터가 나옴
        # ( 63개의 랜드마크 데이터(각 landmark x,y,z 좌표), 21개의 랜드마크 visibility, 15개의 손가락 각도, 1개의 angle_label))
                    action_result = np.concatenate([joint.flatten(), angle_label])
                    
                    # data에 저장
                    data.append(action_result)

                    # 이미지에 landmark 그리기
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            
            #영상저장
            out.write(img)
            cv2.imshow('img', img)

            # q 버튼 누르면 종료
            if cv2.waitKey(1) == ord('q'):
                break
        
        # data에 저장 후 
        # raw data save(=np 형태로), raw data를 합쳐서 cv2.ml.KNearest_create()으로 사용할 수 있다.
        # raw data 각 행의 마지막 index = label
        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('new_dataset', f'raw_{action}_{created_time}'), data)

        # LSTM seq window 생성, 30개의 데이터를 보고 다음을 예측
        # seq_length = 30
        seq_data = []
        for seq in range(len(data) - seq_length): # data - 30
            seq_data.append(data[seq: seq+seq_length]) # data[0: 30] ....

        # seq_data np.array
        seq_data = np.array(seq_data)
        print(action, seq_data.shape)
        # [data개수, seq, len(data)]
        # seq_data save
        np.save(os.path.join('new_dataset', f'seq_{action}_{created_time}'), seq_data)
    break

    '''
    rock (901, 100)
    rock (871, 30, 100)
    scissors (902, 100)
    scissors (872, 30, 100)
    paper (893, 100)
    paper (863, 30, 100)
    '''
