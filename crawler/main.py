import urllib.request
import json
import csv
import codecs

#爬取json数据，需要模拟headers，否则Error 403
headers = {'X-Requested-With': 'XMLHttpRequest',
           'Referer': 'https://xueqiu.com/hq/screener/CN#category=SH&orderby=tweet7d&order=desc&page=1&tweet7d=ALL',
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
           'Host': 'xueqiu.com',
           #'Connection':'keep-alive',
           #'Accept':'*/*',
           'cookie':'s=ff11wkq5h6; u=251491362783476; xq_a_token=ca292f8d934efc28f3fd052b7dcf46f14a20a0d3; xq_r_token=d1accc7b0cafd743be1b975a863a146e514d9c80; __utmt=1; __utma=1.401757876.1491394845.1491450072.1491482853.4; __utmb=1.2.10.1491482853; __utmc=1; __utmz=1.1491394845.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); Hm_lvt_1db88642e346389874251b5a1eded6e3=1491394991,1491431537,1491450072,1491482854; Hm_lpvt_1db88642e346389874251b5a1eded6e3=1491482879'
           }
url = 'https://xueqiu.com/stock/screener/screen.json?category=SH&orderby=tweet7d&order=desc&page=1&tweet7d=ALL'
req = urllib.request.Request(url,headers=headers)
html = urllib.request.urlopen(req).read().decode('utf-8')
#print(html)

#保留需要的变量
data = json.loads(html)
data0 = data['list']
data1 = []
for i in range(0,20):
    row=data0[i]
    row={'No':i+1, 'Symbol':row['symbol'],'Exchange':row['exchange'], 'Name':row['name'], 'Tweet7d':row['tweet7d']}
    data1.append(row)
#print(data1[0])

#写入csv文件
filename='xueqiu.csv'
with open(filename, 'wb') as outf:
    outf.write(codecs.BOM_UTF8)
with open(filename, 'a',newline='', encoding='UTF8') as outf:
    dw = csv.DictWriter(outf,fieldnames=data1[0].keys())
    dw.writeheader()
    for row in data1:
        dw.writerow(row)