#!/usr/bin/env python3
"""Serve local visualization files with byte-range support for MP4 seeking."""

from __future__ import annotations

import argparse
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class RangeRequestHandler(SimpleHTTPRequestHandler):
    range: tuple[int, int] | None = None

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):  # noqa: ANN001
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        if not os.path.exists(path):
            self.send_error(404, "File not found")
            return None

        ctype = self.guess_type(path)
        file_obj = open(path, "rb")
        size = os.fstat(file_obj.fileno()).st_size
        range_header = self.headers.get("Range")
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                start_s, end_s = match.groups()
                if start_s:
                    start = int(start_s)
                    end = int(end_s) if end_s else size - 1
                else:
                    suffix_len = int(end_s)
                    start = max(0, size - suffix_len)
                    end = size - 1
                start = max(0, min(start, size - 1))
                end = max(start, min(end, size - 1))
                self.range = (start, end)
                file_obj.seek(start)
                self.send_response(206)
                self.send_header("Content-type", ctype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("Last-Modified", self.date_time_string(os.path.getmtime(path)))
                self.end_headers()
                return file_obj

        self.range = None
        self.send_response(200)
        self.send_header("Content-type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Last-Modified", self.date_time_string(os.path.getmtime(path)))
        self.end_headers()
        return file_obj

    def copyfile(self, source, outputfile) -> None:  # noqa: ANN001
        if self.range is None:
            return super().copyfile(source, outputfile)
        start, end = self.range
        remaining = end - start + 1
        while remaining > 0:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/Users/zikangjiang/dev/ego-exo"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    os.chdir(args.root)
    server = ThreadingHTTPServer((args.host, args.port), RangeRequestHandler)
    print(f"Serving {args.root} on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
