"""Flask application hosting the neural network
"""
import json
from classes.network import Network
from nnfs.datasets import spiral_data

from flask import Flask
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse
app = Flask(__name__)
CORS(app)
api = Api(app)

args = reqparse.RequestParser()
args.add_argument("name", type=int, help="Name of the person")
args.add_argument("count", type=int, help="Count of the person")

class DisplayNetwork(Resource):
    def get(self):
        with open("schema.json", "r") as f:
            schema = json.load(f)
        return {"network": schema}

api.add_resource(DisplayNetwork, "/network")

class Display(Resource):
    def get(self, name):    
        return {"name": f"Hello {name}"}, 200

    def put(self, val):
        x = args.parse_args()

# api.add_resource(Display, "/network/<string:name>")

if __name__ == "__main__":
    app.run(debug=True)