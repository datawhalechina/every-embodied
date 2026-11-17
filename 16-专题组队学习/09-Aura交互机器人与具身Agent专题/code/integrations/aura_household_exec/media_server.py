"""Own loopback-only, read-only static media service with single HTTP ranges."""
import argparse,functools,json,re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path


class MediaHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.byte_range=None
        path=Path(self.translate_path(self.path)).resolve()
        if not path.is_relative_to(Path(self.directory).resolve()):
            self.send_error(HTTPStatus.FORBIDDEN);return None
        requested=self.headers.get('Range')
        if not requested or not path.is_file():return super().send_head()
        size=path.stat().st_size;match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested.strip())
        if not match or not any(match.groups()):return self.range_error(size)
        first,last=match.groups()
        if first:
            start=int(first);end=min(size-1,int(last)) if last else size-1
        else:
            count=int(last)
            if count==0:return self.range_error(size)
            start=max(0,size-count);end=size-1
        if start>=size or end<start:return self.range_error(size)
        stream=path.open('rb');stream.seek(start);self.byte_range=(start,end)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header('Content-type',self.guess_type(str(path)))
        self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length',str(end-start+1))
        self.send_header('Last-Modified',self.date_time_string(path.stat().st_mtime))
        self.end_headers();return stream

    def range_error(self,size):
        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        self.send_header('Content-Range',f'bytes */{size}')
        self.send_header('Content-Length','0');self.end_headers();return None

    def end_headers(self):
        self.send_header('Accept-Ranges','bytes')
        self.send_header('Access-Control-Allow-Origin','*')
        super().end_headers()

    def copyfile(self,source,outputfile):
        if self.byte_range is None:return super().copyfile(source,outputfile)
        remaining=self.byte_range[1]-self.byte_range[0]+1
        while remaining:
            chunk=source.read(min(256*1024,remaining))
            if not chunk:break
            outputfile.write(chunk);remaining-=len(chunk)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--port',type=int,default=8799)
    args=parser.parse_args();directory=args.directory.resolve()
    if not directory.is_dir():raise ValueError('Existing selected public artifact directory required')
    handler=functools.partial(MediaHandler,directory=str(directory))
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler)
    print(json.dumps({'bind':'127.0.0.1','port':args.port,'directory':str(directory),'http_ranges':True,'read_only':True}),flush=True)
    server.serve_forever()

if __name__=='__main__':main()
