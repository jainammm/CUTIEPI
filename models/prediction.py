import os
import tensorflow as tf
import numpy as np

from models.loaddata import DataLoader
from configs.config import model_params
from models.CUTIEv1 import CUTIERes as CUTIEv1
from models.utils import vis_bbox

c_threshold = 0.5

def predict(json_data):
    data_loader = DataLoader(model_params, json_data, update_dict=False, load_dictionary=True) # yaha pe params update karne hai

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

        data = data_loader.fetch_validation_data()
        feed_dict = {
            network.data_grid: data['grid_table'],
        }

        fetches = [model_output]

        [model_output_val] = sess.run(fetches=fetches, feed_dict=feed_dict)
        
        shape = data['shape']
        file_name = data['file_name'][0] # use one single file_name
        bboxes = data['bboxes'][file_name]
        vis_bbox(data_loader, './sample', np.array(data['grid_table'])[0], 
                        np.array(data['gt_classes'])[0], np.array(model_output_val)[0], file_name, 
                        np.array(bboxes), shape)

        model_output_val = np.array(model_output_val)[0]
        logits = model_output_val.reshape([-1, data_loader.num_classes])

        grid_table = np.array(data['grid_table'])[0] 
        gt_classes = np.array(data['gt_classes'])[0]
        file_name = data['file_name'][0]
        text_ids = data['text_ids'][file_name]

        data_input_flat = grid_table.reshape([-1])
        labels = gt_classes.reshape([-1])

        
        for i in range(len(data_input_flat)):
            if max(logits[i]) > c_threshold:
                inf_id = np.argmax(logits[i])
                if inf_id:
                    try:
                        print('----------')
                        print(data_loader.classes[inf_id])
                        print(idTotext(text_ids[i], json_data))
                        print(max(logits[i]))
                        # print(labels[i])
                        # print(data_input_flat[i])
                    except:
                        pass

        
def idTotext(id, json_data):
    for text_box in json_data['text_boxes']:
        if text_box['id'] == id:
            return text_box['text']
    
    return None
    