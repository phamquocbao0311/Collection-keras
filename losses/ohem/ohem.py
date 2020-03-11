def ohem_loss(ytrue, ypred):
    result = K.categorical_crossentropy(ytrue, ypred)
    alpha = K.variable(0.1, dtype=tf.float32)
    index = K.greater(result, alpha)
    return K.mean(result[index])
