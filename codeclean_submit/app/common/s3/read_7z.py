import io

import py7zr
from py7zr.compressor import SevenZipDecompressor


class SevenZipReadStream(io.IOBase):
    def __init__(self, stream):
        self.stream = stream
        self.z = py7zr.SevenZipFile(stream)

        self.f = None
        for f in self.z.files:
            if not f.is_directory:
                self.f = f
                break

        if self.f is None:
            raise Exception("cannot find file in 7z stream.")

        uncompressed_size: int = f.uncompressed  # type: ignore
        compressed_size = f.compressed
        folder = f.folder

        assert uncompressed_size is not None
        assert compressed_size is not None
        assert folder is not None

        # self.decompressor = folder.get_decompressor(compressed_size)
        self.decompressor = SevenZipDecompressor(
            folder.coders,
            compressed_size,
            folder.unpacksizes,
            folder.crc,
            folder.password,
            1 << 20,  # blocksize=1MiB
        )

        self.size = uncompressed_size
        self.pos = 0

    def readable(self):
        return True

    def read(self, n=None):
        if self.pos >= self.size:
            return b""

        rem = self.size - self.pos
        sz = rem if n is None else min(rem, n)

        chunk = self.decompressor.decompress(self.z.fp, sz)
        self.pos += len(chunk)

        return bytes(chunk)

    def tell(self):
        return self.pos

    def close(self):
        self.z.close()
