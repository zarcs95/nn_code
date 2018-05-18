import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code
x_entrenamiento = x_data[:105]
y_entrenamiento = y_data[:105]
x_validacion = x_data[106:128]
y_validacion = y_data[106:128]
x_test = x_data[129:150]
y_test = y_data[129:150]

import matplotlib.cm as cm
import matplotlib.pyplot as plt

print ("\nSome samples...")
for i in range(20):
    print (x_data[i], " -> ", y_data[i])
print


x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)


"""
print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 20
mal_clasificado = 0
for epoch in range(100):
    for jj in range(int(len(x_data) / batch_size)):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    print ("Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}))
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        if np.argmax(b) != np.argmax(r):
            mal_clasificado = mal_clasificado + 1
        print (b, "-->", r)
    print ("----------------------------------------------------------------------------------")
print (mal_clasificado)


"""





print ("----------------------")
print ("   Start training...  ")
print ("----------------------")
#np.argmax(v) -> funcion que devuelve el máximo argumento de un vector
batch_size = 20
mal_clasificado = 0
table = []
for epoch in range(150):
    for jj in range(int(len(x_entrenamiento) / batch_size)):
        batch_xs = x_entrenamiento[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_entrenamiento[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    errores = sess.run(loss, feed_dict={x: x_validacion, y_: y_validacion})
    table.append(errores)

    print ("Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: x_validacion, y_: y_validacion}))
    result = sess.run(y, feed_dict={x: x_test})
for b, r in zip(y_test, result):
    if np.argmax(b) != np.argmax(r):
        mal_clasificado = mal_clasificado + 1
    print (b, "-->", r)
    print ("----------------------------------------------------------------------------------")
print (mal_clasificado)
plt.plot(table)
plt.show()


""" Como podemos apreciar a la hora de contar los mal clasificados, vemos que esi entrenamos con un 70%
de los datos, validamos con un 15% y testeamos con el otro 15% clasifica correctamente casi la totalidad 
de los datos, siendo en número máximo de fallos en 2 mientras que con el otro pasan de 300"""