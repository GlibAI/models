# coding=utf-8
import os
from numpy.random import choice
import shutil
import tensorflow as tf
import syntaxnet.load_parser_ops
from syntaxnet import sentence_pb2, task_spec_pb2
from tensorflow.python.platform import gfile
from google.protobuf import text_format

class glibUtils(object):

    def __init__(self):

        syntaxnet_root_dir = '/home/mohit/glib-tensorflow/models/syntaxnet'
        self.maps_context = {
            'en' : os.path.join(syntaxnet_root_dir, 'syntaxnet/models/parsey_mcparseface/context.pbtxt'),
            'no' : os.path.join(syntaxnet_root_dir, 'syntaxnet/models/parsey_universal/context.pbtxt'),
            'hi' : os.path.join(syntaxnet_root_dir, 'syntaxnet/models/parsey_universal/context.pbtxt'),
        }

        self.maps_tokenizer_context = {
            'en' : os.path.join(syntaxnet_root_dir, 'syntaxnet/models/parsey_universal/context.pbtxt'),
            'no' : os.path.join(syntaxnet_root_dir, 'syntaxnet/models/parsey_universal/context.pbtxt'),
            'hi' : os.path.join(syntaxnet_root_dir, 'syntaxnet/models/parsey_universal/context.pbtxt'),
        }

        self.maps_model = {
            'en' : os.path.join(syntaxnet_root_dir, 'syntaxnet/models/parsey_mcparseface'),
            'no' : os.path.join(syntaxnet_root_dir, '../../syntaxnet_models/Norwegian'),
            'hi' : os.path.join(syntaxnet_root_dir, '../../syntaxnet_models/Hindi'),
        }

        self.maps_tokenizer_model = {
            'en' : os.path.join(syntaxnet_root_dir, '../../syntaxnet_models/English'),
            'no' : os.path.join(syntaxnet_root_dir, '../../syntaxnet_models/Norwegian'),
            'hi' : os.path.join(syntaxnet_root_dir, '../../syntaxnet_models/Hindi')
        }

        self.syntaxnet_root_dir = syntaxnet_root_dir

    def get_root_dir(self):
        return self.syntaxnet_root_dir

    def remove_files(self, tmp_dir, id):
        path = os.path.join(tmp_dir, id + '.pbtxt')
        if os.path.exists(path):
            os.remove(path)

        path = os.path.join(tmp_dir, id + '_raw.conll')
        if os.path.exists(path):
            os.remove(path)

        path = os.path.join(tmp_dir, id + '_tokenized.conll')
        if os.path.exists(path):
            os.remove(path)

        path = os.path.join(tmp_dir, id + '_tagged.conll')
        if os.path.exists(path):
            os.remove(path)

        path = os.path.join(tmp_dir, id + '_parsed.conll')
        if os.path.exists(path):
            os.remove(path)

    def get_tmp_dir(self):
        tmp_dir = os.path.join(self.syntaxnet_root_dir, 'glib_files')
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        return tmp_dir

    def get_context(self, lang):
        return self.maps_context[lang]

    def get_tokenizer_context(self, lang):
        return self.maps_tokenizer_context[lang]

    def get_model_dir(self, lang):
        return self.maps_model[lang]

    def get_tokenizer_model_dir(self, lang):
        return self.maps_tokenizer_model[lang]

    def generate_id(self):
        return str(''.join(str(choice(range(0,9))) for i in range(12)))

    def copy_context_file(self, src, dest_dir, id):
        dest = os.path.join(dest_dir, id + '.pbtxt')
        shutil.copy(src, dest)
        return dest

    def update_context(self, context_file, model_dir, files_dir, id):

        def add_input(name, file_pattern, record_format, cxt):
            inp = cxt.input.add()
            inp.name = name
            inp.record_format.append(record_format)
            inp.part.add().file_pattern = file_pattern

        context = task_spec_pb2.TaskSpec()
        with gfile.FastGFile(context_file) as fin:
            text_format.Merge(fin.read(), context)
        for resource in context.input:
            for part in resource.part:
                if part.file_pattern != '-':
                    part.file_pattern = os.path.join(model_dir, part.file_pattern)

        add_input('raw', os.path.join(files_dir, id + '_raw.conll'), 'untokenized-text', context)
        add_input('tokenized', os.path.join(files_dir, id + '_tokenized.conll'), 'conll-sentence', context)
        add_input('tagged', os.path.join(files_dir, id + '_tagged.conll'), 'conll-sentence', context)
        add_input('parsed', os.path.join(files_dir, id + '_parsed.conll'), 'conll-sentence', context)

        with open(context_file, 'wb') as fout:
            fout.write(str(context))
        return fout.name

    def write_raw(self, files_dir, id, text):
        file_path = os.path.join(files_dir, id + '_raw.conll')
        with open(file_path, 'wb') as fi:
            fi.write(text.encode('utf-8') + '\n')

    def deserialize(self, document):
        sentence = sentence_pb2.Sentence()
        sentence.ParseFromString(document)
        return sentence

    def build_response(self, query, tokenized, tagged, parsed):
        tokens = []
        text = query
        assert len(parsed.token) == len(tagged.token) == len(tokenized.token)
        k = 0
        for token, tag, parse in zip(tokenized.token, tagged.token, parsed.token):
            word = token.word
            start = token.start
            stop = token.end
            coarse = tag.category
            if not coarse:
                coarse, fine = tag.tag.split('++')
                fine = fine if fine else ''
            else:
                fine = tag.tag
            label = parse.label.upper()
            if label != 'ROOT':
                head = parse.head
            else:
                head = k
            k += 1
            tmp = {}
            tmp[u'text'] = {}
            tmp[u'text'][u'content'] = word.encode('utf-8')
            tmp[u'text'][u'beginOffset'] = start
            tmp[u'lemma'] = u''
            tmp[u'partOfSpeech'] = {}
            tmp[u'partOfSpeech'][u'tag'] = coarse.encode('utf-8')
            tmp[u'partOfSpeech'][u'fine'] = fine.encode('utf-8')
            tmp[u'dependencyEdge'] = {}
            tmp[u'dependencyEdge'][u'headTokenIndex'] = head
            tmp[u'dependencyEdge'][u'label'] = label.encode('utf-8')
            tokens.append(tmp)
        return tokens
