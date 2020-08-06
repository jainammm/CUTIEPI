import os
import tensorflow as tf

trained_checkpoint_prefix = 'pretrained_model/CUTIE.ckpt'
export_dir = os.path.join('model_for_serving', 'CUTIE', '1')

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    # Restore from checkpoint
    print('jainmam')
    output_node_names = ['softmax']
    loader = tf.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    print(frozen_graph_def)

    

    # Export checkpoint to SavedModel
    # prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
    #     {"input": inputs}, {"output": output})

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.TRAINING,
                                             tf.saved_model.tag_constants.SERVING],
                                         #  signature_def_map={
                                         #      "classification": prediction_signature},
                                         strip_default_attrs=True)
    builder.save()
