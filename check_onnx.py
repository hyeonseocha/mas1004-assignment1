import urllib.request
url = 'https://hyeonseocha.github.io/mas1004-assignment1/model.onnx'
try:
    req = urllib.request.Request(url, method='HEAD')
    r = urllib.request.urlopen(req)
    print(f'Status: {r.status}')
    print(f'Content-Length: {r.headers.get("Content-Length")}')
    print(f'Content-Type: {r.headers.get("Content-Type")}')
    print(f'ETag: {r.headers.get("ETag")}')
except Exception as e:
    print(f'Error: {e}')