import tensorflow as tf
import itertools
import numpy as np

class OneDirection():
    def __init__(self,**kwargs):
        print("\napproach: ", "one_direction\n")

    def transform(self,z,z_norm,y,step_sizes,**kwargs):

        dim_z = z.get_shape().as_list()[1]
        with tf.variable_scope("parameters_to_train"):
            self.w = tf.get_variable("w_ganalyze", [1, dim_z], dtype=tf.float32, initializer=tf.zeros_initializer)

        z_new = z
        z_new = z_new + step_sizes * self.w
        z_new_norm = tf.expand_dims(tf.norm(z_new, axis=1), axis=1)
        z_new = z_norm * z_new / z_new_norm

        return (z_new)

    def compute_loss(self, current, target):
        loss = tf.reduce_mean(tf.square(current - target))

        return loss

    def loss_to_file(self,feed_dict,batch_start,lossfile,sess):
        overall_loss_copy =  tf.get_default_graph().get_tensor_by_name("loss_ganalyze:0")
        overall_loss_eval = sess.run(overall_loss_copy,feed_dict=feed_dict)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start) + ",overall_loss," + str(overall_loss_eval)+"\n")

