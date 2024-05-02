import tensorflow as tf

# Define a constant tensor
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

# Create a TensorFlow session and specify GPU device
with tf.device('/GPU:0'):
    # Start a TensorFlow session
    with tf.Session() as sess:
        # Run the session to evaluate the tensor
        result = sess.run(tensor)
        print(result)