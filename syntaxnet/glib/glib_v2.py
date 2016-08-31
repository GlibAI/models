import os
import shutil
import time
import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from syntaxnet.ops import gen_parser_ops
from syntaxnet import structured_graph_builder
from syntaxnet import sentence_pb2


class Analyzer(object):
    def __init__(self):
        model_dir="/home/mohit/glib-tensorflow/models/syntaxnet/glib/parsey_mcparseface"
        task_context="%s/context.pbtxt" % model_dir

        common_params = {
            "task_context":  task_context,
            "beam_size":     8,
            "max_steps":     1000,
            "graph_builder": "structured",
            "batch_size":    1024,
            "slim_model":    True,
        }

        self.model = {
            "brain_tagger": {
                "arg_prefix":         "brain_tagger",
                "hidden_layer_sizes": "64",
                "input":              None,
                "model_path":         "%s/tagger-params" % model_dir,

            },
            "brain_parser": {
                "arg_prefix":         "brain_parser",
                "hidden_layer_sizes": "512,512",
                "input":              None,
                "model_path":         "%s/parser-params" % model_dir,
            },
        }

        for prefix in ["brain_tagger","brain_parser"]:
            self.model[prefix].update(common_params)
            feature_sizes, domain_sizes, embedding_dims, num_actions = self.get_feature_size(task_context, prefix)
            self.model[prefix].update({'feature_sizes': feature_sizes,
                                  'domain_sizes': domain_sizes,
                                  'embedding_dims': embedding_dims,
                                  'num_actions': num_actions })

        session = tf.Session()
        text_input = tf.placeholder(tf.string, [None])
        document_source = gen_parser_ops.document_source(text=text_input,
                                                         task_context=task_context,
                                                         corpus_name="stdin",
                                                         batch_size=common_params['batch_size'],
                                                         documents_from_input=True)

        for prefix in ["brain_tagger","brain_parser"]:
            with tf.variable_scope(prefix):
                if True or prefix == "brain_tagger":
                    source = document_source.documents if prefix == "brain_tagger" else self.model["brain_tagger"]["documents"]
                    self.model[prefix]["documents"] = self.build_graph(session, source, self.model[prefix])

        self.session = session
        self.text_input = text_input

    def build_graph(self, sess, document_source, FLAGS):
        task_context = FLAGS["task_context"]
        arg_prefix = FLAGS["arg_prefix"]
        num_actions = FLAGS["num_actions"]
        feature_sizes = FLAGS["feature_sizes"]
        domain_sizes = FLAGS["domain_sizes"]
        embedding_dims = FLAGS["embedding_dims"]
        hidden_layer_sizes = map(int, FLAGS["hidden_layer_sizes"].split(','))
        beam_size = FLAGS["beam_size"]
        max_steps = FLAGS["max_steps"]
        batch_size = FLAGS["batch_size"]
        corpus_name = FLAGS["input"]
        slim_model = FLAGS["slim_model"]
        model_path = FLAGS["model_path"]

        parser = structured_graph_builder.StructuredGraphBuilder(
            num_actions,
            feature_sizes,
            domain_sizes,
            embedding_dims,
            hidden_layer_sizes,
            gate_gradients=True,
            arg_prefix=arg_prefix,
            beam_size=beam_size,
            max_steps=max_steps)

        parser.AddEvaluation(task_context,
                             batch_size,
                             corpus_name=corpus_name,
                             evaluation_max_steps=max_steps,
                             document_source=document_source)

        parser.AddSaver(slim_model)
        sess.run(parser.inits.values())
        parser.saver.restore(sess, model_path)

        return parser.evaluation['documents']

    def get_feature_size(self, task_context, arg_prefix):
        with tf.variable_scope("fs_"+arg_prefix):
            with tf.Session() as sess:
                return sess.run(gen_parser_ops.feature_size(task_context=task_context,
                                                            arg_prefix=arg_prefix))

    def analyze(self, query):
        d = self.session.run(self.model["brain_parser"]["documents"], feed_dict={self.text_input: [query]})
        if len(d) > 0:
            d = d[0]
            sent = sentence_pb2.Sentence()
            sent.ParseFromString(d)
            return sent
        else:
            return None

