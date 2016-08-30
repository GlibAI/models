# coding=utf-8
import os
import tensorflow as tf
import utils as glib_utils
import tokenizer as glib_tokenizer
import tagger as glib_tagger
import parser as glib_parser
import time
import numpy as np


class Glib(object):
    def __init__(self):
        self.utils = glib_utils.glibUtils()
        self.tmp_dir = self.utils.get_tmp_dir()

    def analyze(self, lang, query):
        utils = self.utils
        tmp_dir = self.tmp_dir

        token_context = utils.get_tokenizer_context(lang)
        token_model_dir = utils.get_tokenizer_model_dir(lang)

        id = utils.generate_id()
        context = utils.copy_context_file(token_context, tmp_dir, id)
        context = utils.update_context(context, token_model_dir, tmp_dir, id)
        utils.write_raw(tmp_dir, id, query)

        tokenizer = glib_tokenizer.Tokenizer()
        tokenizer_results = tokenizer.get(context, token_model_dir, tmp_dir, id)
        if tokenizer_results is not None:
            tagger = glib_tagger.Tagger()
            if lang != 'en':
                model_dir = token_model_dir
            else:
                context = utils.get_context(lang)
                model_dir = utils.get_model_dir(lang)
                context = utils.copy_context_file(context, tmp_dir, id)
                context = utils.update_context(context, model_dir, tmp_dir, id)

            tagger_results = tagger.get(context, model_dir, tmp_dir, id)
            if tagger_results:
                parser = glib_parser.Parser()
                parser_results = parser.get(context, model_dir, tmp_dir, id)
                if parser_results:
                    result = utils.build_response(query, tokenizer_results, tagger_results, parser_results)
                    utils.remove_files(tmp_dir, id)
                    return result
                else:
                    utils.remove_files(tmp_dir, id)
                    return None
            else:
                utils.remove_files(tmp_dir, id)
                return None
        else:
            utils.remove_files(tmp_dir, id)
            return None


