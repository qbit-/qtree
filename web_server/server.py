import src.web_api as api
import json
import pickle

from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
graph = None


@app.route("/")
def hello():
    return "Qtree server program"


@app.route("/parse_graph", methods=('GET', 'POST'))
def parse_graph():
    print(request)
    if request.method == "POST":
        contents = request.form.get('inst')
        if contents:
            with open('tmp.gr', 'w+') as f:
                f.write(contents)
            filename = 'tmp.gr'
        else:
            filename = request.form.get('inst_name')
        if filename:
            print("reading file", filename)
            graph = api.read_graph_from_circfile(filename)
            g = graph
            # g = api.copy_to_simple(graph)
            pickle.dump(g, open('graph.p', 'wb'))
            print(g.edges())
            print('nodes', g.nodes())

            json = api.graph_to_d3json(g)
            return json

        return "Not yet impl,use filename", 501
    else:
        return "use POST", 400


@app.route("/eliminate", methods=('GET',))
def eliminate():
    print(request)
    if request.method == "GET":
        contents = request.args.get('vertex_id')
        print(request.args)
        if contents:
            node = int(contents)
            print("eliminating", node)

            graph = pickle.load(open('graph.p', 'rb'))
            # graph = api.copy_to_simple(graph)
            api.eliminate_node(graph, node)
            pickle.dump(graph, open('graph.p', 'wb'))

            json = api.graph_to_d3json(graph)
            print(graph.edges(), graph.nodes())
            print(len(list(graph.edges())))
            return json


@app.route("/node_info", methods=('GET',))
def node_info():
    print(request)
    if request.method == "GET":
        contents = request.args.get('vertex_id')
        print(request.args)
        if contents:
            node = int(contents)
            print("getting info", node)
            graph = pickle.load(open('graph.p', 'rb'))
            expr = api.get_contraction_expression(graph, node)
            costs = api.get_cost_by_node(graph, node)
            mem, flops = costs
            res = {
                'expr': expr,
                'mem': mem,
                'flops': flops
            }
            return json.dumps(res)

        return "Not yet implemented,use filename", 501
    else:
        return "use GET", 400


if __name__ == "__main__":
    app.run(host='0.0.0.0')
