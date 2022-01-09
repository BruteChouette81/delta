from flask import Flask, json, jsonify
from flask import request
import pprint
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import select

#import uuid

#from bot import strat_reader
 
# Flask Constructor
app = Flask(__name__)
app.config ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user.sqlite3'


db = SQLAlchemy(app)
class user(db.Model):
  id = db.Column('user_id', db.Integer, primary_key = True)
  name = db.Column(db.String(100))
  password = db.Column(db.Integer)
  hash = db.Column(db.String(100))

  def __init__(self, name, password, hash):
    self.name = name
    self.password = password
    self.hash = hash

#db.create_all()

@app.route("/new_user", methods = ['GET', 'POST'])
def new_user():
  if request.method == 'POST':
    jsonData = request.get_json(force=True)
    pprint.pprint(jsonData)
    hash = "010d0nenc" #just for test
    new_user = user(jsonData["name"], jsonData["pass"], hash)

    db.session.add(new_user)
    db.session.commit()

    positive = {"positive": True}

    return jsonify(positive)

  else:
    return "hello world"
 
@app.route("/login", methods = ['POST'])
def login():
  if request.method == 'POST':
    jsonData = request.get_json(force=True)
    log_user = db.session.execute(select(user).filter_by(name=jsonData["name"])).scalar_one()
    good_pass = log_user.password

    if good_pass == jsonData["pass"]:
      return jsonify({"log": True})

    else:
      return jsonify({"log": False})
  

if __name__ == "__main__":
  #db.create_all()
  app.run(host="0.0.0.0")