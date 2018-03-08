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
  - Loss Function
          ![equation](http://latex.codecogs.com/gif.latex?L%28w%2Cb%29%3D%5Cfrac%7B1%7D%7B2n%7D%7B%7C%7CXw%2Bb-Y%7C%7C%7D%5E2)
  - Defining loss function
    ```python          
    # Model Parameters
    W = tf.get_variable("W", shape=(d), initializer=tf.random_normal_initializer)                                                                                                 
    b = tf.get_variable("b", shape=(1), initializer=tf.random_normal_initializer)                                                                                                 
                                  
    # Data Placeholders input to the Graph
    X = tf.placeholder(tf.float32)                                                                                   
    Y = tf.placeholder(tf.float32)                                                                                                                                                
                       
    # Loss Function
    linear_model = tf.reduce_sum(tf.multiply(X,W), axis=1) + b                                                                                                                    
    sqaured_deltas = tf.square(linear_model - Y)                                                                                                                                  
    loss = (1.0/(2*n))*tf.reduce_sum(sqaured_deltas)                                                                                                                                
    ```
  - Training
    ```python
    # Define optimizer
    optimizer = tf.train.AdamOptimizer(0.003) #tf.train.GradientDescentOptimizer(0.003)                                                                                                                       
    train = optimizer.minimize(loss)
    
    with tf.Session(config=config) as sess:                                                                                                                                       
      # Initialization
      init = tf.global_variables_initializer()                                                                                                                                  
      sess.run(init)  
    # Train Iteratively
      for i in range(100):                                                                                                                                                      
        _, l = sess.run([train,loss], feed_dict={X:data['X'], Y:data['Y']})                                                                                                   

    ```
## 4. Logistic Regression
  - Loss function
    ```python
       scores = tf.sigmoid(tf.reduce_sum(tf.multiply(X,params['W']), axis=1) + params['b'])                                                                                      
       loss = tf.reduce_sum(tf.log(1.0 + tf.exp(-1.0*tf.multiply(scores, Y)))) 
    ```

## 5. Simple Neural Network
  - Loss function
    ```python
    # Store layers weight & bias
    weights = {                                                                                                                                                                       
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),                                                                                                                   
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),                                                                                                                
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))                                                                                                                 
    }                                                                                                                                                                                 
    biases = {                                                                                                                                                                        
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),                                                                                                                            
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),                                                                                                                            
    'out': tf.Variable(tf.random_normal([n_classes]))                                                                                                                             
    }                                                                                                                                                                                 
    # Create model                                                                                                                                                                    
    def multilayer_perceptron(x):                                                                                                                                                     
      # Hidden fully connected layer with 256 neurons                                                                                                                               
      layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])                                                                                                                   
      # Hidden fully connected layer with 256 neurons                                                                                                                               
      layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])                                                                                                             
      # Output fully connected layer with a neuron for each class                                                                                                                   
      out_layer = tf.matmul(layer_2, weights['out']) + biases['out']                                                                                                                
      return out_layer                                                                                                                                                              
                                                                                                                                                                                  
    # Construct model                                                                                                                                                                 
    logits = multilayer_perceptron(X)                                                                                                                                                 
                                                                                                                                                                                  
    # Define loss and optimizer                                                                                                                                                       
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(                                                                                                                 
    logits=logits, labels=Y))                                                                                                                                                     

    ```
  - Batch Training
    ```python
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      train_op = optimizer.minimize(loss_op)
      # Initializing the variables
      init = tf.global_variables_initializer()

      with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
          avg_cost = 0.
          total_batch = int(mnist.train.num_examples/batch_size)
          # Loop over all batches
          for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        print("Optimization Finished!")

        # Test model
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    ```
  - Simpler way
    ```python
      # Define the neural network
      def neural_net(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.layers.dense(x, n_hidden_1, name="layer_1") #default name="dense"
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.layers.dense(layer_1, n_hidden_2, name="layer_2") #default name="dense_1"
        # Output fully connected layer with a neuron for each class
        out_layer = tf.layers.dense(layer_2, num_classes, name="layer_3") #default name="dense_2"
        return out_layer
    ```
    - How to get parameters ?
    ```python
      model.get_variable_names()
      #['layer_1/bias', 'layer_1/kernel', 'layer_2/bias', 'layer_2/kernel', 'layer_3/bias', 'layer_3/kernel', 'global_step']
      model.get_variable_value("layer_1/kernel")
      # prints the corresponding weights
    ```
    - Alternate way of training (using Estimator)
    ```python
      # Build the Estimator
      model = tf.estimator.Estimator(model_fn) # model_fn(features, labels, mode) : EstimatorSpec

      # Define the input function for training
      input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'images': mnist.train.images}, y=mnist.train.labels,
      batch_size=batch_size, num_epochs=None, shuffle=True)
      # Train the Model
      model.train(input_fn, steps=num_steps)

      # Evaluate the Model
      # Define the input function for evaluating
      input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'images': mnist.test.images}, y=mnist.test.labels,
      batch_size=batch_size, shuffle=False)
      # Use the Estimator 'evaluate' method
      e = model.evaluate(input_fn)

      print("Testing Accuracy:", e['accuracy'])
    ```
    - Estimator specifications
    ```python
      # Define the model function (following TF Estimator Template)
      def model_fn(features, labels, mode):
        # Build the neural network
        logits = neural_net(features)

        # Predictions
        pred_classes = tf.argmax(logits, axis=1)
         pred_probas = tf.nn.softmax(logits)

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
          return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op,
          global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=pred_classes,
          loss=loss_op,
          train_op=train_op,
          eval_metric_ops={'accuracy': acc_op})

        return estim_specs
      ```
<!-- 
![equation](http://latex.codecogs.com/gif.latex?Concentration%3D%5Cfrac%7BTotalTemplate%7D%7BTotalVolume%7D)  
-->

