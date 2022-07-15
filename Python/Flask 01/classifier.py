from flask import Flask, request

app = Flask(__name__)

@app.route("/classify", methods=['GET', 'POST'])
def classify():
    pl = request.args.get('pl')
    pw = request.args.get('pw')
    sl = request.args.get('sl')
    sw = request.args.get('sw')
    # result = xgboost_classify_iris(pl, pw, sl, sw)
    return "result"



if __name__ == '__main__':
    app.run()