# BluetoothPathLoss
RSSI to Distance의 형태를 Neural Network 형태로 문제를 해결하기 위한 프로젝트
regression 형태로 문제를 해결하고 있음
지도학습이기 때문에 데이터 수집이 필수

## 논문 구현
### 제목 : 저전력 무선 통신용 인공신경망 기반 경로 손실 모델
- Artificial Neural Network based Path Loss Model for Low-Power Wireless Communication
- 김성현, 문성우, 김대겸, 고명진, 최용훈
- 2021 한국통신학회 발표

## 데이터셋 구조 (v1)
Type|Description|Unit
----|------|-----
MAC|Bluetooth MAC Addres|Address
METER|Rx와 Tx가 떨어진 거리를 의미|meter
RSSI|신호 세기 (Rx가 수신하는)|dBm
TXPOWER|Tx의 송출 세기득|dBm
TXHEIGHT|Tx의 높이|meter
RXHEIGHT|Rx의 높이|meter
TX_ANTENNA_GAIN|Tx안테나 이득|dBi
RX_ANTENNA_GAIN|Rx안테나 게인|dBi
FSPL|Free Space Path Loss 계산 값|meter
environment|환경 변수 (장애물, 날씨 등을 뜻함)|constant

## 데이터셋 구조 (v2)
Type|Description|Unit
----|------|-----
MAC|Bluetooth MAC Addres|Address
METER|Rx와 Tx가 떨어진 거리를 의미|meter
RSSI|신호 세기 (Rx가 수신하는)|dBm
TXPOWER|Tx의 송출 세기득|dBm
TXHEIGHT|Tx의 높이|meter
RXHEIGHT|Rx의 높이|meter
TX_ANTENNA_GAIN|Tx안테나 이득|dBi
RX_ANTENNA_GAIN|Rx안테나 게인|dBi
FSPL|Free Space Path Loss 계산 값|meter
Covered|안테나를 둘러싸고 있는 장애물|constant
Adertising Channel|데이터를 수집한 채널 정보|constant: 37,38,39

## 사용한 모델
- DNN : Fully Connected Layer 4계층으로 이루어진 모델 (Dropout layer 존재, Activiation Function은 PReLU 사용)
- LSTM : Recurrent 한 모델을 적용해보기위해 LSTM 사용 (many to one의 형태로 구현)
- CRNN : Convolution Layer와 LSTM 계층이 합쳐진 구조 사용
## V1 실험 모델
- dataset : non-scaled dataset
