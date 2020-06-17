import web
from segment.seg_and_tag import *

class DictHandle(object):
    def GET(self):
        return "hello world"
    def POST(self):
        webdata = web.input()
        textdata = webdata.get('text')