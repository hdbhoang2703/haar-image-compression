import numpy as np
import cv2
from scipy.sparse import csc_matrix
from PIL import Image
import io

class HaarCompressor:
    def __init__(self, image_file, threshold):
        self.image_file = image_file  # BytesIO object
        self.threshold = threshold
        self.img = self._read_image()
        self.N = self.img.shape[0]
        self.H = self._haar_matrix(self.N)
        self.transformed = []
        self.compressed = []
        self.original_size = None
        self.sparse_size = None
        self.non_zero_elements = None
        self.total_elements = None

    def _read_image(self):
        """ Đọc ảnh từ BytesIO và resize về hình vuông kích thước 2^n """
        self.image_file.seek(0)
        file_bytes = np.asarray(bytearray(self.image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Không thể đọc ảnh.")

        size = int(2 ** np.floor(np.log2(max(img.shape[:2]))))
        img = cv2.resize(img, (size, size))
        return img.astype(float)

    def _haar_basis_function(self, N, k):
        h = np.zeros(N)
        if k == 0:
            h[:] = 1 / np.sqrt(N)
        else:
            p = int(np.floor(np.log2(k)))
            q = k - 2**p + 1
            step = 2**p
            width = N // step
            start = (q - 1) * width
            mid = start + width // 2
            end = start + width
            h[start:mid] = np.sqrt(step / N)
            h[mid:end] = -np.sqrt(step / N)
        return h

    def _haar_matrix(self, N):
        H = np.zeros((N, N))
        for k in range(N):
            H[k, :] = self._haar_basis_function(N, k)
        return H

    def transform(self):
        self.transformed = [
            self.H @ self.img[:, :, i] @ self.H.T for i in range(3)
        ]

    def compress(self):
        if not self.transformed:
            self.transform()
        self.compressed = []
        self.non_zero_elements = 0
        self.total_elements = self.N * self.N * 3
        sparse_total_bytes = 0

        for c in self.transformed:
            compressed = np.where(np.abs(c) < self.threshold, 0, c)
            sparse = csc_matrix(compressed)
            self.compressed.append(sparse)
            self.non_zero_elements += sparse.count_nonzero()
            sparse_total_bytes += (
                sparse.data.nbytes + sparse.indices.nbytes + sparse.indptr.nbytes
            )

        self.original_size = self.img.nbytes / 1024
        self.sparse_size = sparse_total_bytes / 1024

    def decompress(self):
        if not self.compressed:
            raise ValueError("Ảnh chưa được nén.")
        channels = []
        for c in self.compressed:
            dense = c.toarray()
            rec = self.H.T @ dense @ self.H
            rec = np.clip(rec, 0, 255).astype(np.uint8)
            channels.append(rec)
        return np.stack(channels, axis=2)

    def get_compressed_image_bytes(self):
        """
        Trả ảnh nén dưới dạng BytesIO để hiển thị hoặc tải về.
        """
        img_decompressed = self.decompress()
        pil_img = Image.fromarray(cv2.cvtColor(img_decompressed, cv2.COLOR_BGR2RGB))

        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")  # Lưu đúng định dạng PNG
        buffer.seek(0)
        return buffer

