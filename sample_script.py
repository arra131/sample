import tensorflow as tf

# Check if GPU is available
if tf.test.is_gpu_available():
    # Define a constant tensor
    tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

    # Create a TensorFlow session and specify GPU device
    with tf.device('/GPU:0'):
        # Start a TensorFlow session
        with tf.Session() as sess:
            # Run the session to evaluate the tensor
            result = sess.run(tensor)
            print(result)

            # Check the device placement of the tensor
            print("Tensor is placed on device:", tensor.device)
else:
    # Print error message if GPU is not available
    raise RuntimeError("GPU not available. Please run this code on a machine with GPU support.")
