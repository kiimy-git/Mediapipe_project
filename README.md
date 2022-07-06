# Movement Detection using Mediapipe
[Mediapipe solutions hands API](https://google.github.io/mediapipe/solutions/hands.html)

### 목표
OpenCV, Mediapipe API를 활용하여 특정 행동에 대한 데이터 수집 및 label 부여하고 행동 예측 모델 구현 및 성능을 파악한다.

### 문제 인식
현재 자동화, 자율 주행과 같은 인공지능 기술이 많이 화두가 되고 있으며 관련 기술 개발 활발

특히 자율주행 부분에서 사람은 가장 큰 장애물 중 하나일 수 있다.

도로에서 보다 안전한 자율 주행을 위해서는 주변의 운전자, 자전거 타는 사람 및 보행자 등이 다음에 무슨 행동을 할 것 인가에 대해 예측할 수 있어야 한다.

### Benefit
해당 프로젝트는 Hand detection만 진행했지만 더 나아가 사람 몸을 탐지할 수 있는 Body detection을 통해서 일어날 수 있는 

사고에 대한 데이터들을 수집 데이터 학습을 통해 예측 및 임계값을 부여하여 주의를 감지할 수 있게 적용한다면 위험 요소를 사전에 방지하며 보다 안전한 사회를 만들 수 있을 것


# Requirements
[requirements](https://github.com/kimmy-git/Mediapipe_project/blob/main/requirements.txt)

## Tools
* python
* numpy
* openCV
* **Mediapipe**
* Tensorflow, Keras, LSTM
* os, time
* sklearn

# Process
- [Data](#data)
  + [Mediapipe Hands API](#mediapipe-hands-api)
* [Model](#model)
* [Metrics and Score](#metrics-and-score)
  + [Train Process](#train-process)
* [Reviews](#reviews)

# Data
[create_dataset.py](https://github.com/kimmy-git/Mediapipe_project/blob/main/create_dataset.py)
* webcam을 사용하여 데이터 수집(gestures = rock, scissors, paper)
* Data = Mediapipe solution API (Hands)를 활용하여 손의 각 Landmark의 각도를 구한 값(**Arccos사용**)
* (rock, scissors, paper) = 2606개 수집
```python
'''
  rock (901, 100)

  scissors (902, 100)

  paper (893, 100)
'''
```
### Mediapipe Hands API
![initial](https://user-images.githubusercontent.com/83389640/144199605-62ff7b8d-cea2-4293-bd47-18cf26b0dcff.png)

### Arccos ( cosine 역함수 )
<p align="center"><img src="https://user-images.githubusercontent.com/83389640/144199611-84a5739a-db5f-4c7e-8181-de0bc6922e4a.png"></p>

# Model
* 딥러닝 기반 모델 : LSTM
* 머신러닝 기반 모델(**StandardScaler**) : LogisticRegression, RidgeClassifier, **RandomForestClassifier**, XGBClassifier 

# Metrics and Score
* Accuracy

# Train Process
1. Webcam으로 Data 수집([new_dataset](https://github.com/kimmy-git/Mediapipe_project/tree/main/new_dataset) = .npy파일) = sequence설정 data, raw data 저장
2. sequence Model LSTM, ML model(RandomForest 사용) = 해당 데이터 학습
3. Modeling
4. 학습시킨 Model을 사용하여 webcam으로 실행 후 결과 확인

## 1. Data 수집(Mediapipe 사용하여 Hands의 각 Landmark 각도 구하기)

```python
mp_hands = mp.solutions.hands # 손의 관절 위치를 인식할 수 있는 모델
mp_drawing = mp.solutions.drawing_utils # 관절(landmark 그리기 위함)
hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence=0.5,
    # 1.0에 가까울수록 확실한 확률이 높은 손가락만 감지
    min_tracking_confidence=0.5
    # 1.0에 가까울수록 추적율이 높은 것만 감지
)
'''
* Detection, tracking, recognition의 차이
1. Dectection - 영상에서 대상을 찾는 것은 detection
2. Recognition - 찾은 대상이 무엇인지 식별하는 것
3. Tracking - 비디오 영상에서 특정 대상의 위치 변화를 추적하는 것
'''

# 관절사이 각도 계산(= gesture_train 값)    
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
# einsum 'ij,ij->i'  행별 원소끼리 곱하고 합친것 ==> 1D로 변환
angle = np.arccos(np.einsum('ij,ij->i', A, B)) 
    # [15,] = 15개의 손가락 데이터

# degrees 변경(=gesture_train), float64 ==> float32
# radian * (180*pi)
angle = np.degrees(angle)
angle_label = np.array([angle], dtype=np.float32)

# idx = gestures index(gestures = ['rock', 'scissors', 'paper'])
angle_label = np.append(angle_label, idx)
```
## Data 수집 영상
* rock, scissors, paper 순차적으로 데이터 수집
* ### **Target data = {'0':'rock', '1':'scissors', '2':'paper'}**
<p align="center"><img src="https://user-images.githubusercontent.com/83389640/144220487-bbee0733-d9d7-41f9-bcf4-b9fd00917679.gif"></p>

## 2. colab으로 LSTM Model, ML Model 학습
### 1. LSTM Model
* early_stopping 사용 / 사용 X 두 경우 모델 학습 진행
* Earlystopping 사용시 학습 조기종료로 인해 성능이 100%가 나오지 않음
```python
# LSTM
model = Sequential([
                    LSTM(64, activation='relu', input_shape=X_train.shape[1:3]),
                    Dense(32, activation='relu'),
                    Dense(3, activation='softmax')
])
                    
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
```
<p align="center"><img src="https://user-images.githubusercontent.com/83389640/144205231-282223dc-15ac-4d94-98e6-d1129806be72.png"></p>



```python
#학습한 모델 저장(EarlyStopping)
cheak_point = ModelCheckpoint('LSTM_Model.h5',
                              monitor='val_acc',   # val_acc 값이 개선되었을때 호출
                              verbose=1,            # 로그를 출력
                              save_best_only=True,  # 가장 best 값만 저장
                              mode='auto')           # auto는 알아서 best 찾는다 min/max

early_stop = EarlyStopping(monitor='val_acc',min_delta=0.0001, patience=3)

history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100,
                    callbacks=[cheak_point, early_stop])
                    
------------------------------------------------------------------------------------------------------------------------
Epoch 4/100
74/74 [==============================] - 7s 88ms/step - loss: 1.1787e-07 - acc: 1.0000 - val_loss: 0.0466 - val_acc: 0.9962

Epoch 00004: val_acc did not improve from 0.99617
------------------------------------------------------------------------------------------------------------------------

#학습한 모델 저장(EarlyStopping X)
cheak_point = ModelCheckpoint('LSTM_Model2.h5',
                              monitor='val_acc',   # val_acc 값이 개선되었을때 호출
                              verbose=1,            # 로그를 출력
                              save_best_only=True,  # 가장 best 값만 저장
                              mode='auto')           # auto는 알아서 best 찾는다 min/max

history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=30,
                    callbacks=[cheak_point])
                    
------------------------------------------------------------------------------------------------------------------------                    
Epoch 30/30
74/74 [==============================] - 7s 91ms/step - loss: 0.0162 - acc: 0.9970 - val_loss: 0.0040 - val_acc: 0.9962

Epoch 00030: val_acc did not improve from 1.00000
------------------------------------------------------------------------------------------------------------------------

# multilabel confusion matrix
# EarlyStopping 사용
array([[[171,   0],
        [  1,  89]],

       [[175,   1],
        [  0,  85]],

       [[175,   0],
        [  0,  86]]])
------------------------------------------------------------------------------------------------------------------------
# EarlyStopping X
array([[[171,   0],
        [  0,  90]],

       [[176,   0],
        [  0,  85]],

       [[175,   0],
        [  0,  86]]])
```
### Accuracy, loss 시각화
![initial](https://user-images.githubusercontent.com/83389640/144205395-af9c1dbc-e8c5-4767-bace-2f27ed3fcb8b.png)

### 2. ML Model 학습
- 모델 default 값으로 진행
- 진행한 모델 모두 100% 성능(RandomForest 모델로 진행)
- LogisticRegression, RidgeClassifier, **RandomForestClassifier**, XGBClassifier

### Confusion Matrix
<p align="center"><img src="https://user-images.githubusercontent.com/83389640/144202628-ab80bc49-8665-4763-84ae-4fbb91978fd8.png"></p>


### 3. 학습한 모델(LSTM, RandomForest) 적용하여 결과 확인
(용량이 커서 paper부분은 짤림)

### LSTM(영상에 예측 gestures 이름 추가)
<p align="center"><img src="https://user-images.githubusercontent.com/83389640/144220672-d6b9ce2c-71f6-426b-a00b-8e9087a2a0ec.gif"></p>

### RandomForest(예측확률 값 추가)
<p align="center"><img src="https://user-images.githubusercontent.com/83389640/144220685-1dc862e1-60f2-4816-a993-aecb5e8603bd.gif"></p>


# Reviews
  - scissors는 두 가지 형태로 보여질 수 있는데(= V or 가위 형태) 현재 V형태로 학습
    - 둘 다 똑같은 뜻인데 가위형태는 scissors로 안보고 rock으로 판단
      * 해당 행동은 학습이 되지않았고 확률상 높은 것을 예측
    - 내가 주먹이라고 해서 다른 사람이 똑같이 주먹이라고 보지 않을 것임
      - 결국 해당 행동에 대해서 어떻게 정의를 할 것인가에 대한 고민이 많을 것(= Data Quilty의 중요성)

### 어려웠던 점
  - **Mediapipe - 내부적으로 어떻게 구현했는지에 대한 이해 한계(= 어떻게 손관절을 인식??? 그려지는 라인은 서로 어떻게 그려졌는지???)**
    - 영상 인식 라이브러리(= OpenCV) 적용 어려움
    - 적용된 함수에 필요한 input 값은 무엇인지 output 값은 무엇인지 우선 파악 후 적용
    - 손관절 인식 구현 부분은 이해 한계 = 영상 인식에 대한 기초적인 학습이 필요
      - [Github - OpenCV_Tutorial](https://github.com/kimmy-git/OpenCV-Tutorial)
      - [Blog](https://cord-ai.tistory.com/category/%2A%20OpenCV)
  - **코사인 유사도 적용(= 구현)**
    - 코사인 유사도, 아크코사인 활용 용도 파악
    - 구현하기 위한 코딩역량 부족
      - [코딩 학습]() 

### 느낀점
  - **Model 성능 100%???** 
    - scissors가 다른 action에 비해서 예측확률이 현저히 떨어짐 why???
      - **data split 문제**
        * paper가 test_data로 들어가 있음, 각 클래스마다 data split 진행 후 concate 방식으로 진행해야함
        
      ==> 예측확률을 추가적으로 확인함으로써 성능에 의존하면 위험성이 존재할 수 있음
      
      ==> Data split 후 추가적으로 타겟에 대한 비율 확인 필요성 확인
      
  - Mediapipe 모델 구현 및 성능을 하는데 목적을 둠
    - 선택된 모델(=RF, LSTM) 직접 테스트를 통해 대체적으로 다 좋은 성능을 보임
    
    ==> 모델의 특성을 떠나서 Mediapipe API를 통한 손 관절을 제대로 인식, 이를 수치화한 데이터가 명확했기 때문이라고 생각
   

### 개선점
  - 예측확률이 낮은데 예측한 행동은 정확한가???
    * 예측확률이 현저히 낮을 때는 임계값을 설정(= 특정 행동에 대한 이상치 탐지용으로 적용 가능)
  - Hands 이외에도 Pose, Face Detection or 특정 부분을 Segmentation하는 다양한 API가 존재함
    * Data 수집시 해당 API를 활용하면 유용할 
