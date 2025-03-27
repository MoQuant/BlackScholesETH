def BuildDataSet():
    import websocket
    import json
    import time
    import datetime
    import pandas as pd

    def Cyclone(name):
        coin, expiry, strike, optype = name.split('-')
        T0 = int(time.time())
        T1 = time.mktime(datetime.datetime.strptime(expiry, '%d%b%y').timetuple())
        return float(strike), (T1 - T0)/(60*60*24*365), optype

    conn = websocket.create_connection('wss://test.deribit.com/ws/api/v2')
    msg = {"jsonrpc": "2.0",
         "method": "public/get_index_price",
         "id": 42,
         "params": {
            "index_name": "eth_usd"}
        }
    conn.send(json.dumps(msg))
    resp = json.loads(conn.recv())
    S = resp['result']['index_price']
    rf = 0

    msg = {
          "method" : "public/get_instruments",
          "params" : {
            "currency" : "ETH",
            "kind" : "option",
            "exipred": False
          },
          "jsonrpc" : "2.0",
          "id" : 1
        }
    conn.send(json.dumps(msg))
    resp = json.loads(conn.recv())

    cols = ['S','K','r','v','t','premium','optype']
    dataset = []

    for i in resp['result']:
        iname = i['instrument_name']
        print(iname)
        K, T, optype = Cyclone(iname)
        msg = {
          "jsonrpc" : "2.0",
          "id" : 3659,
          "method" : "public/get_book_summary_by_instrument",
          "params" : {
            "instrument_name" : iname
          }
        }
        conn.send(json.dumps(msg))
        resp = json.loads(conn.recv())
        v = resp['result'][0]['mark_iv']
        premium = resp['result'][0]['mark_price']

        dataset.append([S, K, rf, v, T, premium, optype])

    df = pd.DataFrame(dataset, columns=cols)
    df.to_csv('Ethereum.csv')
    print("Data has loaded")

def AnalyzeDataSet():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)          # No line wrapping
    pd.set_option('display.max_colwidth', None)

    N = lambda x: norm.cdf(x)

    def Regression(x, y):
        cov = np.cov(x, y)
        beta = cov[0, 1] / cov[0, 0]
        return beta

    def PutScholes(S, K, r, v, t):
        d1 = (np.log(S/K) + (r + 0.5*pow(v, 2))*t)/(v*np.sqrt(t))
        d2 = d1 - v*np.sqrt(t)
        return K*np.exp(-r*t)*N(-d2) - S*N(-d1)

    data = pd.read_csv('Ethereum.csv')
    del data['Unnamed: 0']

    data['premium'] *= data['S']
    data['v'] /= 100.0

    data = data.sample(frac=1).reset_index(drop=True)

    puts = data[(data['optype'] == 'P')]

    bs = []
    reg = []
    for S, K, r, v, t, premium, o in puts.values.tolist():
        bsprice = PutScholes(S, K, r, v, t)
        bs.append(bsprice)
        reg.append(premium)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(bs, reg, color='red')
    ax.set_title('Option Price Beta: {}'.format(round(Regression(bs, reg), 6)))
    ax.set_xlabel('Black Scholes Price')
    ax.set_ylabel('Exchange Price')
    
    plt.show()


AnalyzeDataSet()
