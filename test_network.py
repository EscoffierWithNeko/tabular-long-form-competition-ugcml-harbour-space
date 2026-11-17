import urllib.request
import socket

ports = [7890, 7897, 10808, 10809, 1080, 8080]
found = None

for port in ports:
    proxy_url = f'http://127.0.0.1:{port}'
    try:
        handler = urllib.request.ProxyHandler({'http': proxy_url, 'https': proxy_url})
        opener = urllib.request.build_opener(handler)
        req = urllib.request.Request('https://api.kaggle.com', headers={'User-Agent': 'Mozilla/5.0'})
        with opener.open(req, timeout=3) as resp:
            print(f'Proxy {proxy_url} works! Status: {resp.status}')
            found = proxy_url
            break
    except Exception as e:
        # print(f'Proxy {proxy_url} error: {e}')
        pass

if not found:
    print('Testing direct connection to api.kaggle.com...')
    try:
        req = urllib.request.Request('https://api.kaggle.com', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f'Direct connection works! Status: {resp.status}')
    except Exception as e:
        print(f'Direct connection failed: {e}')
