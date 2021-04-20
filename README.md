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
