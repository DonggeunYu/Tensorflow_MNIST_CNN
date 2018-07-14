# Tensorflow_MNIST_CNN

 ## Summary

Python의 Tensorflow를 활용하여 MNIST 데이터 셋을 학습시키는 것이 목표이다.

이미지 학습에 뛰어난 성능을 가진 CNN(Convolution Neural Network)를 활용할 것이다.



## MNIST Data

MNIST는 사람들의 손글씨가 그려져 있는 데이터이다. MNIST의 데이터는 28x28x1의 형태로 주어지며 Train과 Test 데이터로 나뉘게 된다. Train과 Test에는 라벨이 달려있어 그 손글씨가 어떤 숫자인지를 알려준다. Train 데이터는 학습에 사용할 계획이고 Test 데이터는 학습 후에 테스트를 해보는 용도로 사용할 것이다.

![](https://raw.githubusercontent.com/Yudonggeun/Tensorflow_MNIST_CNN/master/Image/Image1.png)



## Source Code

### Iibrary

tensorflow를 사용할 것이다.

MNIST를 사용하기 위해서 tensorflow의 예제를 이용한다.

```Python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
```



### Function

shape으로 배열 모양의 입력을 받으면 truncated_normal 함수로 절단정규분포의 난수로 배열을 채워서 상수로 만들어준다.

```Python
def weight_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
    return tf.Variable(initial)
```

``` Python
def bias_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
    return tf.Variable(initial)
```



 합성곱을 해주는 함수이다.tf.nn.conv2d를 사용하여 (1, 1)칸씩을 이동하면서 x와 w를  합성곱 해주고  padding 설정을 통해 출력은 입력한 사진 즉 28x28의 크기로 출력을 해준다. 그리고 마지막으로 bias를 더해준다.

``` Python
def conv2d(x, w, bias):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + bias
```



relu함수는 0보다 작은 값은 0으로 0보다 큰 값은 그대로 출력한다.

![](https://raw.githubusercontent.com/Yudonggeun/Tensorflow_MNIST_CNN/master/Image/Image2.png)

```Python
def relu(x):
    return tf.nn.relu(x)
```