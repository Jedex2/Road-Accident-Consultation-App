# import flast module
from flask import Flask

# instance of flask application
app = Flask(__name__)

# home route that returns below text when root url is accessed
@app.route("/")

def hello_world():
    return "<p>Wellcome To ROAD ACCIDENT CONSULTATION</p>"

@app.route("/about")
def about():
    return "<h1>เราเป็น 1 ใน ตัวช่วยในกระบวนการตัดสินใจ ไม่ว่าคุณจะรู้สึกว่า ประกันของคู่กรณีหรือว่าปม้แต่ตํารวจเอง ทําให้คุณรู้สึกเสียเปรียบ เราสามารถช่วยคุณวิเคราะห์ สถานะการณ์เบื้องต้น โดยอิงตามกฎจราจร<h1>"
if __name__ == '__main__':  
   app.run()  
