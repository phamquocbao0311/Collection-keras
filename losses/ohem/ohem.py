def ohem_loss(ytrue, ypred):
    result = K.categorical_crossentropy(ytrue, ypred)
    sorted_result = tf.sort(result,axis = -1, direction = "DESCENDING")
    alpha = K.variable(0.1, dtype=tf.float32)
    index = K.greater(result, alpha)
    # loss = tf.multiply(result, index)
    return K.mean(result[index])
