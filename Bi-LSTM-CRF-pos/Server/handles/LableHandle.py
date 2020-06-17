import web
from segment.seg_and_tag import *

class LableHandle(object):
    def POST(self):
        webdata = web.input()
        textdata = webdata.get('text')