import os
import shutil
import time
import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from syntaxnet.ops import gen_parser_ops
from syntaxnet import structured_graph_builder
from syntaxnet import sentence_pb2

def Build(sess, document_source, FLAGS):
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


def GetFeatureSize(task_context, arg_prefix):
    with tf.variable_scope("fs_"+arg_prefix):
        with tf.Session() as sess:
            return sess.run(gen_parser_ops.feature_size(task_context=task_context,
                                                        arg_prefix=arg_prefix))


def main():
    logging.set_verbosity(logging.ERROR)

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

    model = {
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
        model[prefix].update(common_params)
        feature_sizes, domain_sizes, embedding_dims, num_actions = GetFeatureSize(task_context, prefix)
        model[prefix].update({'feature_sizes': feature_sizes,
                              'domain_sizes': domain_sizes,
                              'embedding_dims': embedding_dims,
                              'num_actions': num_actions })

    with tf.Session() as sess:
        runs = 100
        sentences = ['I want a pizza',
                     'What is your name?',
                     'Can you show me your menu please!',
                     'I asked for delivery, but now you are asking me to pick it up from your restaurant; that is ridiculous!']

        text_input = tf.placeholder(tf.string, [None])
        # text_input = tf.constant(["parsey is the greatest"], tf.string)

        document_source = gen_parser_ops.document_source(text=text_input,
                                                         task_context=task_context,
                                                         corpus_name="stdin",
                                                         batch_size=common_params['batch_size'],
                                                         documents_from_input=True)

        for prefix in ["brain_tagger","brain_parser"]:
            with tf.variable_scope(prefix):
                if True or prefix == "brain_tagger":
                    source = document_source.documents if prefix == "brain_tagger" else model["brain_tagger"]["documents"]
                    model[prefix]["documents"] = Build(sess, source, model[prefix])

        # sink = gen_parser_ops.document_sink(model["brain_parser"]["documents"],
        #                                     task_context=task_context,
        #                                     corpus_name="stdout-conll")

        t1 = time.time()
        num_docs = 0
        for run in range(runs):
            for s in sentences:
                d = sess.run(model["brain_parser"]["documents"], feed_dict={text_input: [s]})
                if len(d) > 0:
                    d = d[0]
                    sent = sentence_pb2.Sentence()
                    sent.ParseFromString(d)
                    print sent
                num_docs += 1
        t2 = time.time()
        print 'Docs Processed: ' + str(num_docs)
        print 'Total time: ' + str(t2 - t1)

if __name__ == '__main__':
    main()