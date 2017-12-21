## SpeechRecognitionChallenge

음성인식챌린지를 위한 코드입니다 <br\>

convert_wav로 wav파일을 npy로 따로 만들어 npy 파일을 이용합니다<br\>
npy파일이 데이터를 읽어오는데 시간이 빠르기때문에 변환해서 사용합니다<br\>
npy파일로 불러올 때도 scipy의 라이브러리 함수로 샘플링이 되기 때문에
필요에 따라 조정이 필요합니다<br\>
plot을 그려보시면, 전처리가 필요한 것을 알 수 있습니다<br\>
  
label.csv는 train의 클래스를 0~29까지 적어둔 파일입니다<br\>
  
모델부분은 김성훈교수님의 강의의 코드와 아래 블로그를 참고하였습니다<br\>
https://danijar.com/variable-sequence-lengths-in-tensorflow/
