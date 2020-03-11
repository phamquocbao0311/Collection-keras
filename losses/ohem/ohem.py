def ohem_loss(ytrue, ypred):
    result = keras.losses.categorical_crossentropy(ytrue, ypred, label_smoothing=0)
    alpha = K.variable(0.1, dtype=tf.float32)
    index = K.greater(result, alpha)
    return K.mean(result[index])
