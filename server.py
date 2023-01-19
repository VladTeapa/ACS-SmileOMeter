import http.server
import socketserver
import io
import cgi
import cv2
import base64
import json
import sys
sys.path.insert(0, './deepface-fer')
sys.path.insert(0, './CNN')
from main import get_voting_based
PORT = 44444
class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def modify_image(self,img):
        for i in range(100,200):
            for j in range(100,200):
                img[i][j][0] = 255;
                img[i][j][1] = 255;
                img[i][j][2] = 255;
        return img
    def do_POST(self):        
        r, info = self.deal_post_data()
        print(r, info, "by: ", self.client_address)
        f = io.BytesIO()
        img = cv2.imread('image')
        #img = self.modify_image(img)
        imgResized = img
        #imgResized = cv2.resize(img, (48,48))
        cv2.imwrite('temp.jpeg', cv2.resize(img,(48,48)))
        res = get_voting_based('temp.jpeg')
        
        retval, buffer = cv2.imencode('.jpeg',img)
        jpg_as_text = base64.b64encode(buffer)
        if r:
            f.write(bytes('{"data":"',"utf-8"))
            f.write(jpg_as_text)
            f.write(bytes('","emotions":'+json.dumps(res),"utf-8"))
            f.write(bytes('}',"utf-8"))
        else:
            f.write(b"Failed\n")
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(length))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if f:
            self.copyfile(f, self.wfile)
            f.close()      

    def deal_post_data(self):
        ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
        pdict['CONTENT-LENGTH'] = int(self.headers['Content-Length'])
        if ctype == 'multipart/form-data':
            form = cgi.FieldStorage( fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD':'POST', 'CONTENT_TYPE':self.headers['Content-Type'], })
            print (type(form))
            try:
                if isinstance(form["file"], list):
                    for record in form["file"]:
                        open("./%s"%record.filename, "wb").write(record.file.read())
                else:
                    open("./%s"%form["file"].filename, "wb").write(form["file"].file.read())
            except IOError:
                    return (False, "Can't create file to write, do you have permission to write?")
        return (True, "Files uploaded")

Handler = CustomHTTPRequestHandler
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()