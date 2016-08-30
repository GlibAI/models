# coding=utf-8
import os
import json
import cherrypy, time
import numpy as np
import glib
import netifaces as ni


class Root:
    def __init__(self):
        self.analyzer = glib.Glib()

    @cherrypy.expose
    def index(self, **params):
        start = time.time()
        if 'lang' in params:
            lang = params['lang']
        else:
            lang = 'en'
        if 'q' in params:
            query = params['q']
        else:
            query = 'I want a pizza by 6 tomorrow.'

        results = self.analyzer.analyze(lang, query)
        stop = time.time()
        total_time = np.round(stop - start, 2)
        if results:
            ret = {'success': 'OK', 'results': results, 'time': total_time}
            return json.dumps(ret)
        else:
            ret = {'success': 'FAIL', 'time': total_time}
            return json.dumps(ret)


def main():
    ip = ni.ifaddresses('eth0')[2][0]['addr']
    cherrypy.config.update({'server.socket_host': ip,
                            'server.socket_port': 7204,
                            'server.thread_pool': 30})
    conf = {}
    cherrypy.tree.mount(Root(), '/', conf)
    cherrypy.engine.start()
    cherrypy.engine.block()

if __name__ == '__main__':
    main()
