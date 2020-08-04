import os
import tensorflow as tf

from config.config import model_params

def predict():
    data_loader = DataLoader(model_params, update_dict=False, load_dictionary=True) # yaha pe params update karne hai

    num_words = max(20000, data_loader.num_words)

    num_classes = data_loader.num_classes

    # model
    network = CUTIEv1(num_words, num_classes, model_params)       # yaha oe model_params dekhna padega

    model_output = network.get_output('softmax')

    ckpt_saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            ckpt_path = os.path.join(model_params.ckpt_path, model_params.ckpt_file)
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            print('Restoring from {}...'.format(ckpt_path))
            ckpt_saver.restore(sess, ckpt_path)
            print('{} restored'.format(ckpt_path))
        except:
            raise Exception('Check your pretrained {:s}'.format(ckpt_path))

        
