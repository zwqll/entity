import web
urls = (
    '/segment', 'Handle',
    '/reloaddict', 'DictHandle',
)

class Handle(object):
    def GET(self):
        return "hello world"
    def POST(self):
        webdata = web.input()
        textdata = webdata.get('text')      
        print textdata

if __name__ == '__main__':
    app = web.application(urls, globals())
    app.run()