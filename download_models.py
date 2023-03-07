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

modelset=set()
def download(targets):
    for url in targets:
        # print(url)
        model_name=os.path.relpath(url, 'http://10-91-242-212.iotg.sclab.intel.com/cv_bench_cache/try_builds_cache/').split('/')[1]
        # print(model_name)
        modelset.add(model_name)
        wget_cmd=f"wget -r --no-parent --reject index.html* -c -N {url}"
        # os.system(wget_cmd)
    
download(scrape("http://conformance.sclab.intel.com/neo4j/models/lookupmodels?opname=NonMaxSuppression&opset=5"))
download(scrape("http://conformance.sclab.intel.com/neo4j/models/lookupmodels?opname=NonMaxSuppression&opset=9"))
download(scrape("http://conformance.sclab.intel.com/neo4j/models/lookupmodels?opname=MulticlassNms&opset=9"))
download(scrape("http://conformance.sclab.intel.com/neo4j/models/lookupmodels?opname=MatrixNms&opset=8"))

models=list(modelset)
models.sort()
print("\n".join(models))
print("\nThere are total ", len(modelset), " models.")