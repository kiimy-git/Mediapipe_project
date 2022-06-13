# Movement Detection using Mediapipe
[Mediapipe solutions hands API](https://google.github.io/mediapipe/solutions/hands.html)
* 건설현장에서는 자재 도난사고, 인명사고 등 빈번하게 발생
* 이를 예방하기 위한 CCTV와 관리자가 있지만 실시간으로 Monitoring 한계가 존재

**특정 행동에 대한 데이터를 수집하고 이를 label을 부여한 Model을 구축함으로써 실시간으로 사고를 예방하기 위함**

# Requirements
[requirements](https://github.com/kimmy-git/Mediapipe_project/blob/main/requirements.txt)

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

# Train Process( **Mediapipe API 활용** )
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


## 3. 학습한 모델(LSTM, RandomForest) 적용하여 결과 확인
(용량이 커서 paper부분은 짤림)

### LSTM(영상에 예측 gestures 이름 추가)
<p align="center"><img src="https://user-images.githubusercontent.com/83389640/144220672-d6b9ce2c-71f6-426b-a00b-8e9087a2a0ec.gif"></p>

### RandomForest(예측확률 값 추가)
<p align="center"><img src="https://user-images.githubusercontent.com/83389640/144220685-1dc862e1-60f2-4816-a993-aecb5e8603bd.gif"></p>

## Tools
* python
* numpy
* openCV
* **Mediapipe**
* Tensorflow, Keras, LSTM
* os, time
* sklearn

## Results
1. 복잡한 구조의 모델을 굳이 사용할 필요가 없었다. **why?**
   - 기본적인 model에서도 충분한 성능을 보인다는 것을 확인
   
2. 다양한 Data를 수집하고 label에 대한 예측확률에 임계치를 설정한다면? - 특정 행동에 대한 위험성을 감지 가능

## Reviews
1. Mediapipe라는 이미 잘 구현된 API를 가져와 구축한 것이기 때문에 내부적으로 어떻게 구현했는지에 대한 이해 한계

   ex) 어떻게 Hands Landmark 인식?

2. scissors가 다른 action에 비해서 예측확률이 현저히 떨어짐, 검증 데이터 셋에서도 100프로의 성능을 보여줌 **why??**
   - data split시 문제(= paper가 test_data로 들어가 있음, 각 클래스마다 data split 진행 후 concate)

3. Mediapipe에는 많은 Solution이 존재(= Object Detection, Pose, Face Mesh ...)
