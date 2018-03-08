# TensorFlow : An Introductory Tutorial

## Contents
  1. Computation Graphs
  2. Sessions
  3. Linear Regression
  4. Logistic Regression
  5. Simple Neural Network
  6. Word2Vec
  7. Recurrent Neural Network
  8. Convolutional Neural Network

## 1. Computation Graphs

  - **Symbolic Variables**
    - tf.Variable : A variable maintains state in the graph across calls to run. Used for storing parameters of models.
    - tf.constant : A constant's value is stored in the graph.
    - tf.placeholder : Placeholder for data. Used for supplying input data to the model.
  - **Operations**
    - tf.add : element-wise addition.
    - tf.multiply : element-wise multiplication.
    - tf.matmul : matrix multiplication
    - tf.reduce_sum, tf.reduce_max, ...
    - tf.nn.embedding_lookup
    
## 2. Sessions
  - A Session object encapsulates the environment in which `Operation` objects are executed, and `Tensor` objects are evaluated.
      ```python
      #Build a graph.
      a = tf.constant(5.0) 
      b = tf.constant(6.0)
      add = tf.add(a,b)
      mul = tf.multiply(a,b)
      pow = tf.pow(a,b)

      # Launch the graph in a session.
      sess = tf.Session()

      # Evaluate the tensors .
      print(sess.run(add))  # >> 11.0
      print(sess.run(mul))  # >> 30.0
      print(sess.run(pow))  # >> 15625.0
      ```
  - A session may own resources, so it should be closed after use.
      ```python
      # Launch the graph in a session.
      sess = tf.Session()

      # Evaluate the tensors .
      print(sess.run(add))  # >> 11.0
      sess.close()
      ```
      OR
      ```python
      # Launch the graph in a session.
      with tf.Session() as sess:
        # Evaluate the tensors .
        print(sess.run(add))  # >> 11.0
      sess.close()
      ```
## 3. Linear Regression
  - Data Matrix `X` with `n` rows (instances) and `d` columns (features).
  - Depedent Variable `Y` with `n` real values.
  - Weight parameter `w` and bias parameter `b`.
  - Loss Function ![equation](http://latex.codecogs.com/gif.latex?L%28w%2Cb%29%3D%5Cfrac%7B1%7D%7B2n%7D%7B%7C%7CXw%2Bb-Y%7C%7C%7D%5E2)
      
#### 
<!-- 
![equation](http://latex.codecogs.com/gif.latex?Concentration%3D%5Cfrac%7BTotalTemplate%7D%7BTotalVolume%7D)  
-->

