# BluetoothPathLoss
RSSI to Distance의 형태를 Neural Network 형태로 문제를 해결하기 위한 프로젝트
regression 형태로 문제를 해결하고 있음
지도학습이기 때문에 데이터 수집이 필수

## 데이터셋 구조
- MAC : 블루투스 Address
- METER : Rx와 Tx가 떨어진 거리를 의미 (Distance)
- RSSI : 신호 세기 (Rx가 수신하는)
- TXPOWER : Tx의 송출 세기
- TXHEIGHT : Tx의 높이
- RXHEIGHT : Rx의 높이
- TX_ANTENNA_GAIN : Tx안테나 게인
- RX_ANTENNA_GAIN : Rx안테나 게인
- FSPL : Free Space Path Loss 계산 값 
- environment : 환경 변수 (장애물, 날씨 등을 뜻함)


## 사용한 모델
- FFNN : Fully Connected Layer 4계층으로 이루어진 모델 (Dropout layer 존재, Activiation Function은 PReLU 사용)
- LSTM : Recurrent 한 모델을 적용해보기위해 LSTM 사용 (many to one의 형태로 구현)

## V1 실험 모델
- dataset : scaled dataset (MinMaxScaler)
### RNN
- model : LSTM
- hidden size : 32
- num layers : 1
- input size : 8
- sequence length : 15
- shuffle : True
- bidirectional : False
- batch first : True
- linear : 32 -> 64
- dropout : 0.3
- activation function : PReLU
- linear : 64 -> 32
- dropout : 0.3
- activation function : PReLU
- linear : 32 -> 1
- learning rate : 0.01
- optimizer : Adam or AdamW
- criterion : MSELoss

### FFNN
- linear : 8 -> 64
- dropout : 0.3
- activation function : PReLU
- linear : 64 -> 128
- dropout : 0.3
- activation function : PReLU
- linear : 128 -> 64
- dropout : 0.3
- activation function : PReLU
- linear : 64 -> 1
- learning rate : 0.001 or 0.01
- optimizer : AdamW or Adam
- criterion : MSELoss

## 프로젝트 구조
### data directory
학습을 위한 데이터를 가공하는 역할을 담당
- data.py : 학습을 위해 추가적인 파라미터를 추가, 데이터 스케일링을 진행 가능 