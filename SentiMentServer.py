from tornado.web import RequestHandler
import tornado
from prediction import pred_emotion
import json

class SentimentHandler(RequestHandler):
    def post(self):
        #print(json.loads(self.request.body))
        audio_file = self.request.files['file'][0]
        original_fname = audio_file['filename']
        input_file = "uploads/" + original_fname
        with open(input_file, 'wb') as output_file:
            output_file.write(audio_file['body'])
        emotion = pred_emotion(input_file)
        self.write({'sentiment': emotion})

class IndexHandler(RequestHandler):
    def get(self):
        self.render("index.html")

if __name__ == "__main__":
    port = '5000'
    application = tornado.web.Application([
        (r"/", IndexHandler),
        (r"/sentiment", SentimentHandler),
        ])

    server = tornado.httpserver.HTTPServer(application, max_buffer_size=167772160)  # 10G
    server.listen(port)
    # application.listen(port)
    print("Server started on port : " + str(port))
    tornado.ioloop.IOLoop.instance().start()


