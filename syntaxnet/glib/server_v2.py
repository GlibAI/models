# coding=utf-8
import os
import json
import cherrypy, time
import numpy as np
import glib_v2
import netifaces as ni
import tornado.ioloop
import tornado.web
from numpy.random import choice


queries = ['I want a pizza',
           'Where can I find a good place to eat?',
           'I am looking for a ticket to London for tomorrow evening.',
           'how are you?',
           'I said it from the beginning that this is not a good idea, wish you would listen to me more often.']


class Root:
    def __init__(self):
        self.analyzer = glib_v2.Analyzer()

    @cherrypy.expose
    def index(self, **params):
        global count
        count += 1
        fi = open('/home/mohit/glib-tensorflow/models/syntaxnet/glib/count.txt', 'wb')
        fi.write(str(count) + '\n')
        fi.close()

        start = time.time()
        if 'q' in params:
            query = params['q']
        else:
            query = 'I want a pizza by 6 tomorrow.'

        results = self.analyzer.analyze(query)
        stop = time.time()
        total_time = np.round(stop - start, 2)
        if results:
            ret = {'success': 'OK', 'time': total_time}
            return json.dumps(ret)
        else:
            ret = {'success': 'FAIL', 'time': total_time}
            return json.dumps(ret)


class MainHandler(tornado.web.RequestHandler):

    analyzer = glib_v2.Analyzer()

    def get(self):
        start = time.time()
        query = choice(queries)
        results = MainHandler.analyzer.analyze(query)
        stop = time.time()
        total_time = np.round(stop - start, 2)
        # print results
        if results:
            ret = {'success': 'OK', 'time': total_time}
            self.write(json.dumps(ret))
        else:
            ret = {'success': 'FAIL', 'time': total_time}
            self.write(json.dumps(ret))

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(7205)
    tornado.ioloop.IOLoop.current().start()

# def main():
#     ip = ni.ifaddresses('eth0')[2][0]['addr']
#     cherrypy.config.update({'server.socket_host': ip,
#                             'server.socket_port': 7205,
#                             'server.thread_pool': 10})
#     conf = {}
#     cherrypy.tree.mount(Root(), '/', conf)
#     cherrypy.engine.start()
#     cherrypy.engine.block()
#
# if __name__ == '__main__':
#     main()
#