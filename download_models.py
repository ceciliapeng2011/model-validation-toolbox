import os
import re

import requests
from bs4 import BeautifulSoup

def scrape(vgm_url):
    html_text = requests.get(vgm_url).text
    soup = BeautifulSoup(html_text, 'html.parser')

    attrs = {
        'href': re.compile(r'dldt$')
    }

    tracks = soup.find_all('a', attrs=attrs, string=re.compile(r'^((?!\().)*$'))
    #print(tracks)

    dldt_urls = []
    if len(list(tracks))>0:
        for track in tracks:
            dldt_url = '{}'.format(track['href'])
            #print(dldt_url)
            dldt_urls.append(dldt_url)
    #print(dldt_urls)
    return dldt_urls

def download(targets):
    for url in targets:
        print(url)
        wget_cmd=f"wget -r --no-parent --reject index.html* -c -N {url}"
        os.system(wget_cmd)
    
download(scrape("http://conformance.sclab.intel.com/neo4j/models/lookupmodels?opname=TensorIterator&opset=1"))
download(scrape("http://conformance.sclab.intel.com/neo4j/models/lookupmodels?opname=Loop&opset=5"))
download(scrape("http://conformance.sclab.intel.com/neo4j/models/lookupmodels?opname=LSTMSequence&opset=5"))
download(scrape("http://conformance.sclab.intel.com/neo4j/models/lookupmodels?opname=ReverseSequence&opset=1"))
download(scrape("http://conformance.sclab.intel.com/neo4j/models/lookupmodels?opname=GRUSequence"))