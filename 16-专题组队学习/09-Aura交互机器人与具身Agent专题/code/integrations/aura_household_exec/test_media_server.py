import functools,tempfile,threading,unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from .media_server import MediaHandler

class RangeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        Path(self.temp.name,'video.mp4').write_bytes(bytes(range(256))*8)
        class Quiet(MediaHandler):
            def log_message(self,*args):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=self.temp.name))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.close);self.url='http://127.0.0.1:'+str(self.server.server_port)+'/video.mp4'
    def close(self):self.server.shutdown();self.thread.join();self.server.server_close()
    def test_explicit_range_is_206_with_only_requested_bytes(self):
        with urlopen(Request(self.url,headers={'Range':'bytes=100-119'})) as response:
            self.assertEqual(response.status,206);self.assertEqual(response.headers['Content-Range'],'bytes 100-119/2048');self.assertEqual(response.read(),bytes(range(100,120)))
    def test_suffix_and_head_ranges(self):
        with urlopen(Request(self.url,headers={'Range':'bytes=-4'})) as response:self.assertEqual(response.read(),bytes(range(252,256)))
        with urlopen(Request(self.url,method='HEAD',headers={'Range':'bytes=8-15'})) as response:
            self.assertEqual(response.status,206);self.assertEqual(response.headers['Content-Length'],'8');self.assertEqual(response.read(),b'')
    def test_unsatisfiable_and_multiple_ranges_are_416(self):
        for value in ['bytes=9999-','bytes=0-10,20-30','bytes=-0']:
            with self.assertRaises(HTTPError) as caught:urlopen(Request(self.url,headers={'Range':value}))
            self.assertEqual(caught.exception.code,416)

if __name__=='__main__':unittest.main()
