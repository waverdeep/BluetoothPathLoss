# BluetoothPathLoss
RSSI to Distance의 형태를 Neural Network 형태로 문제를 해결하기 위한 프로젝트
regression 형태로 문제를 해결하고 있음
지도학습이기 때문에 데이터 수집이 필수

## 논문 구현
### 제목 : 저전력 무선 통신용 인공신경망 기반 경로 손실 모델
- Artificial Neural Network based Path Loss Model for Low-Power Wireless Communication
- 김성현, 문성우, 김대겸, 고명진, 최용훈
- 2021 한국통신학회 발표

## 사용한 모델
- DNN : Dense Layer 4계층으로 이루어진 모델
- LSTM : Recurrent 한 모델을 적용해보기위해 LSTM 사용 (many to one의 형태로 구현)
- CRNN : Convolution Layer와 LSTM 계층이 합쳐진 구조 사용

