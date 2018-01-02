## SpeechRecognitionChallenge

<p>음성인식챌린지를 위한 코드입니다

### scipy
<p>convert_wav로 wav파일을 npy로 따로 만들어 npy 파일을 이용합니다
<p>train에서 background noise는 이 코드에서 사용하지 않습니다
<p>convert_wav시 폴더를 삭제하고 사용해야 합니다
<p>npy파일이 데이터를 읽어오는데 시간이 빠르기때문에 변환해서 사용합니다
<p>wav파일을 불러올 때 scipy의 라이브러리 함수로 샘플링이 되기 때문에
<p>필요에 따라 조정이 필요합니다
<p>plot을 그려보시면, 전처리가 필요한 것을 알 수 있습니다
 
### mfcc
<p>librosa.mfcc를 이용하여 feature를 추출한 후 사용한 코드이며
<p>학습된 모델을 복원하여 test에 적용하는 코드까지 있습니다
<p>label.csv는 train의 클래스를 0~29까지 적어둔 파일입니다
  
<p>모델부분은 김성훈교수님의 강의의 코드와 아래 블로그를 참고하였습니다
<p>https://danijar.com/variable-sequence-lengths-in-tensorflow/
