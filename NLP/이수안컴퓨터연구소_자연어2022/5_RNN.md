

+ pytorch로 시작하는 딥러닝
  + https://wikidocs.net/60690



## 순환신경망



### 활용

+ 시간의 흐름에 따라 변하고, 그 변화가 의미를 갖는 데이터로, 순서가 있는 데이터
  + 시계열(날씨, 주가), 자연어
+ 다른 신경망과는 다르게, **기억**을 가짐





+ feed forward Network vs RNN

+ 순환신경망 구조



+ IMDB 데이터 로드 및 전처리
  + `device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')`





+ 임베딩: `nn.Embedding()`
  + https://wikidocs.net/64779