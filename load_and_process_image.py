##################################################################################################################################################################
#################################### LOAD IN IMAGE AS IMAGE ARRAY ################################################################################################
def load_img(path_to_img):
  max_dim = 666
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float64)

  shape = tf.cast(tf.shape(img)[:-1], tf.float64)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img
##################################################################################################################################################################
#################################### CONVERT TENSOR TO IMAGE FOR DIPLAY ##########################################################################################
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)
##################################################################################################################################################################
#################################### PROCESS INPUT TENSOR FOR IMAGE DISPLAY ######################################################################################
def preprocessInput(inputTensor):
  inputTensor = inputTensor*255.0
  preprocessedInput = tf.keras.applications.vgg19.preprocess_input(inputTensor)
  return preprocessedInput
