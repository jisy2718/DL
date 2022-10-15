



https://www.tensorflow.org/tutorials/text/word2vec 따라하고, 공부하는  markdown ()

https://wikidocs.net/book/2155 를 참고



+ skip-gram은 중심 단어(target word)로부터 주변 단어(context word)를 예측하는 모델

  ![img](https://wikidocs.net/images/page/69141/%EA%B7%B8%EB%A6%BC1-1.PNG)

+ 네거티브 샘플링을 사용하는 Skip-gram(Skip-Gram with Negative Sampling, SGNS) 이하 SGNS는 이와는 다른 접근 방식을 취합니다. SGNS는 다음과 같이 중심 단어와 주변 단어가 모두 입력이 되고, 이 두 단어가 실제로 윈도우 크기 내에 존재하는 이웃 관계인지 그 확률을 예측합니다.

![img](https://wikidocs.net/images/page/69141/%EA%B7%B8%EB%A6%BC1-2.PNG)

+ skip-gram data set과 SGNS의 데이터셋

  ![img](https://wikidocs.net/images/page/69141/%EA%B7%B8%EB%A6%BC3.PNG)

  ![img](https://wikidocs.net/images/page/69141/%EA%B7%B8%EB%A6%BC4.PNG)

  

​			





### (1) 용어 설명

+ **target word**
+ **window** : target word 좌우로 몇 개의 단어를 볼 것인지
+ **context word** : target word에서  window 거리 내에 있는 단어들

+ **A negative sample** is defined as a `(target_word, context_word)` pair such that the `context_word` does not appear in the `window_size` neighborhood of the `target_word`. For the example sentence, these are a few potential negative samples (when `window_size` is `2`).

  ```
  (hot, shimmered)
  (wide, hot)
  (wide, sun)
  ```

  + 네거티브 샘플링은 Word2Vec이 학습 과정에서 전체 단어 집합이 아니라 일부 단어 집합에만 집중할 수 있도록 하는 방법

+ 예시

<img src="https://tensorflow.org/tutorials/text/images/word2vec_skipgram.png" alt="word2vec_skipgrams" style="zoom:67%;" />



+ skip-gram의 training 목적함수는 target word가 given되었을 때, context word일 log probability의 sum을 maximize 하는 것이다. $w_t$가 target word, 나머지가 context word , where `c` is the size of the training context.

  <img src="https://tensorflow.org/tutorials/text/images/word2vec_skipgram_objective.png" alt="word2vec_skipgram_objective" style="zoom:67%;" />

+ The basic skip-gram formulation defines this probability using the softmax function. 

  where *v* and *v'* are target and context vector representations of words and *W* is vocabulary size.

  <img src="https://tensorflow.org/tutorials/text/images/word2vec_full_softmax.png" alt="word2vec_full_softmax" style="zoom:67%;" />





## Setup



### (1) Vectorize an example sentence





### (2) Generate skip-grams from one sentence

+ The `tf.keras.preprocessing.sequence` module provides useful functions that simplify data preparation for word2vec. You can use the `tf.keras.preprocessing.sequence.skipgrams` to generate skip-gram pairs from the `example_sequence` with a given `window_size` from tokens in the range `[0, vocab_size)`.



+ 코드

  ```python
  window_size = 2
  positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
        example_sequence, # [1, 2, 3, 4, 5, 1, 6, 7]
        vocabulary_size=vocab_size,  # vocab_size = len(vocab), vocab = {'<pad>': 0, 'the': 1, 'wide': 2, 'road': 3, 'shimmered': 4, 'in': 5, 'hot': 6, 'sun': 7}
        window_size=window_size,
        negative_samples=0)
  print(len(positive_skip_grams))
  print(positive_skip_grams)
  >>>
  26
  [[5, 6], [4, 2], [1, 5], [3, 1], [6, 5], [4, 1], [4, 3], [7, 1], [1, 7], [7, 6], [5, 4], [2, 1], [6, 1], [3, 4], [5, 1], [2, 3], [1, 3], [1, 6], [4, 5], [1, 2], [2, 4], [5, 3], [3, 2], [6, 7], [3, 5], [1, 4]]
  ```

  ```python
  for target, context in positive_skip_grams[:5]:
    print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")
  >>>
  (5, 6): (in, hot)
  (4, 2): (shimmered, wide)
  (1, 5): (the, in)
  (3, 1): (road, the)
  (6, 5): (hot, in)
  ```



### (5) summary

![word2vec_negative_sampling](https://tensorflow.org/tutorials/text/images/word2vec_negative_sampling.png)