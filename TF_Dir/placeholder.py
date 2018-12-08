import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    with tf.name_scope('PlaceholderExample'):
        x = tf.placeholder(tf.float32, shape=(2, 2), name='x')
        y = tf.matmul(x, x, name='matmul')
        # 必须要赋值才有用
        # 赋值方法，numpy通过feed_dict参数填充
        # with tf.Session() as sess:
        #     print(sess.run(y))

        with tf.Session() as sess:
            rand_array = np.random.rand(2, 2)
            print(sess.run(y, feed_dict={x: rand_array}))

        x = tf.sparse_placeholder(tf.float32)
        y = tf.sparse_reduce_sum(x)

        with tf.Session() as sess:
            indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
            # 设置索引为[3,2,0]和[4,5,1]元素分别为1.0和2.0
            values = np.array([1.0, 2.0], dtype=np.float32)

            # 设置稀疏张量对应的稠密张量形状为[7,9,2]
            shape = np.array([7, 9, 2], dtype=np.int64)

            print(sess.run(y, feed_dict={
                x: tf.SparseTensorValue(indices, values, shape)
            }))

            # 向x填充张量3元组
            print(sess.run(y, feed_dict={x: (indices, values, shape)}))

            # 向x填充Numpy多维数组
            sp = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
            sp_value = sp.eval()
            print(sess.run(y, feed_dict={x: sp_value}))
