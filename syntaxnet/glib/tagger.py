# coding=utf-8
import os
import tensorflow as tf
import syntaxnet.load_parser_ops
from syntaxnet import structured_graph_builder
from syntaxnet.ops import gen_parser_ops
import utils as glib_utils


class Tagger(object):

    def __init__(self):
        self.hidden_layer_sizes = '64'
        self.arg_prefix = 'brain_tagger'
        self.batch_size = 1024
        self.input_name = 'tokenized'
        self.output_name = 'tagged'
        self.max_steps = 1000
        self.beam_size = 8
        self.model_file = 'tagger-params'
        self.utils = glib_utils.glibUtils()

    def get(self, task_context, model_dir, tmp_dir, id):
        doc = None
        with tf.Session() as sess:
            feature_sizes, domain_sizes, embedding_dims, num_actions = sess.run(
                gen_parser_ops.feature_size(task_context=task_context,
                                            arg_prefix=self.arg_prefix))

            hidden_layer_sizes = map(int, self.hidden_layer_sizes.split(','))

            model = structured_graph_builder.StructuredGraphBuilder(num_actions,
                                                                    feature_sizes,
                                                                    domain_sizes,
                                                                    embedding_dims,
                                                                    hidden_layer_sizes,
                                                                    gate_gradients=True,
                                                                    arg_prefix=self.arg_prefix,
                                                                    beam_size=self.beam_size,
                                                                    max_steps=self.max_steps)

            model.AddEvaluation(task_context,
                                 self.batch_size,
                                 corpus_name=self.input_name,
                                 evaluation_max_steps=self.max_steps)

            model.AddSaver(True)
            sess.run(model.inits.values())
            model.saver.restore(sess, os.path.join(model_dir, self.model_file))

            sink_documents = tf.placeholder(tf.string)
            sink = gen_parser_ops.document_sink(sink_documents,
                                                task_context=task_context,
                                                corpus_name=self.output_name)

            num_epochs = None
            while True:
                tf_eval_epochs, tf_eval_metrics, tf_documents = sess.run([
                    model.evaluation['epochs'],
                    model.evaluation['eval_metrics'],
                    model.evaluation['documents'],
                ])

                if len(tf_documents):
                    doc = self.utils.deserialize(tf_documents[0])
                    sess.run(sink, feed_dict={sink_documents: tf_documents})
                    break

                if num_epochs is None:
                    num_epochs = tf_eval_epochs
                elif num_epochs < tf_eval_epochs:
                    break

        tf.reset_default_graph()
        return doc


