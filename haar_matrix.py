import numpy as np
from PIL import Image
import os
import argparse
import sys

class HaarCompressor:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        self.threshold = None
        self.image_path = None
        self.levels = None
        self.target_psnr = None
        self.best_params = None
        self.original_image = None
        self.compressed_image = None
        self.compression_stats = None
        self.last_save_path = None

    def _ensure_numpy_image(self, image_input):
        """Chuyển ảnh thành numpy.ndarray bằng PIL"""
        if isinstance(image_input, np.ndarray):
            return image_input
        elif isinstance(image_input, Image.Image):  # PIL Image
            return np.array(image_input)
        elif isinstance(image_input, str):  # path
            try:
                img = Image.open(image_input).convert("RGB")
                return np.array(img)
            except Exception as e:
                raise ValueError(f"Không thể đọc ảnh từ: {image_input} ({e})")
        else:
            raise ValueError("Không hỗ trợ định dạng ảnh này.")

    def _read_image(self):
        return self._ensure_numpy_image(self.image_path)

    @staticmethod
    def haar_1d(signal):
        factor = 1 / np.sqrt(2)
        avg = (signal[::2] + signal[1::2]) * factor
        diff = (signal[::2] - signal[1::2]) * factor
        return np.concatenate([avg, diff])

    @staticmethod
    def inverse_haar_1d(transformed):
        n = len(transformed)
        factor = 1 / np.sqrt(2)
        avg = transformed[:n//2]
        diff = transformed[n//2:]
        recon = np.zeros(n)
        recon[::2] = (avg + diff) * factor
        recon[1::2] = (avg - diff) * factor
        return recon

    def haar_2d(self, channel):
        h, w = channel.shape
        result = channel.copy().astype(float)
        for _ in range(self.levels):
            for i in range(h):
                result[i, :w] = self.haar_1d(result[i, :w])
            for j in range(w):
                result[:h, j] = self.haar_1d(result[:h, j])
            h //= 2
            w //= 2
        return result

    def inverse_haar_2d(self, transformed):
        H, W = transformed.shape
        result = transformed.copy().astype(float)
        curr_h = H // (2 ** self.levels)
        curr_w = W // (2 ** self.levels)
        for _ in range(self.levels):
            curr_h *= 2
            curr_w *= 2
            for j in range(curr_w):
                result[:curr_h, j] = self.inverse_haar_1d(result[:curr_h, j])
            for i in range(curr_h):
                result[i, :curr_w] = self.inverse_haar_1d(result[i, :curr_w])
        return result

    def pad_to_pow2(self, img):
        h, w = img.shape
        new_h = 1 << int(np.ceil(np.log2(h)))
        new_w = 1 << int(np.ceil(np.log2(w)))
        pad_h = new_h - h
        pad_w = new_w - w
        self.levels = int(np.log2(min(new_h, new_w)))
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')
        return padded, h, w

    def compress_channel(self, chan, threshold):
        padded, h, w = self.pad_to_pow2(chan)
        coeffs = self.haar_2d(padded)
        compressed = coeffs * (np.abs(coeffs) > threshold)
        recon_full = self.inverse_haar_2d(compressed)
        recon = recon_full[:h, :w]
        mse = np.mean((chan - recon) ** 2)
        psnr = (10 * np.log10(255**2 / mse)) if mse > 0 else float('inf')
        zero_pct = 100 * np.sum(compressed == 0) / compressed.size
        return compressed, recon, psnr, zero_pct

    def optimize_threshold(self, img_array):
        if self.threshold is None:
            self.threshold = np.arange(0, 100, 5)

        best = {'threshold': None, 'avg_zero': -1, 'psnr': 0}
        chans = img_array if isinstance(img_array, list) else [img_array]

        for t in self.threshold:
            zeros, psnrs = [], []
            for chan in chans:
                _, _, psnr, zero_pct = self.compress_channel(chan, t)
                psnrs.append(psnr)
                zeros.append(zero_pct)

            avg_psnr = np.mean(psnrs)
            avg_zero = np.mean(zeros)

            if avg_psnr >= self.target_psnr and avg_zero > best['avg_zero']:
                best.update({'threshold': t, 'avg_zero': avg_zero, 'psnr': avg_psnr})

        return best

    def compress_image(self, image_path, target_psnr=30, threshold=None):
        self.image_path = image_path
        self.target_psnr = target_psnr
        self.threshold = threshold
        self.original_image = self._read_image()
        channels = [self.original_image] if self.original_image.ndim == 2 else [self.original_image[:, :, c] for c in range(3)]

        best = self.optimize_threshold(channels)
        if best['threshold'] is None:
            raise ValueError(f"No threshold found to achieve PSNR >= {self.target_psnr} dB.")

        recon_ch, psnr_list, zero_list = [], [], []
        for chan in channels:
            _, recon, psnr, zero_pct = self.compress_channel(chan, best['threshold'])
            recon_ch.append(recon)
            psnr_list.append(psnr)
            zero_list.append(zero_pct)

        recon_img = recon_ch[0] if len(recon_ch) == 1 else np.stack(recon_ch, axis=2)
        self.compressed_image = np.clip(recon_img, 0, 255).astype(np.uint8)
        self.best_params = best
        self.compression_stats = {'psnr_list': psnr_list, 'zero_list': zero_list}
        return self.compressed_image

    def compress_image_by_threshold(self, image_input, threshold):
        self.original_image = self._ensure_numpy_image(image_input)
        channels = [self.original_image] if self.original_image.ndim == 2 else [self.original_image[:, :, c] for c in range(3)]

        recon_ch, psnr_list, zero_list = [], [], []
        for chan in channels:
            _, recon, psnr, zero_pct = self.compress_channel(chan, threshold)
            recon_ch.append(recon)
            psnr_list.append(psnr)
            zero_list.append(zero_pct)

        compressed_image = recon_ch[0] if len(recon_ch) == 1 else np.stack(recon_ch, axis=2)
        self.compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
        self.best_params = {'threshold': threshold}
        self.compression_stats = {'psnr_list': psnr_list, 'zero_list': zero_list}
        return self.compressed_image

    def display_results(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        titles = ['Original', 'Reconstructed']
        images = [self.original_image, self.compressed_image]

        for ax, img, title in zip(axs, images, titles):
            ax.imshow(img.astype(np.uint8))
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def save_reconstructed_image(self):
        name, ext = os.path.splitext(os.path.basename(self.image_path))
        out_dir = self.output_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}_recon{ext}")

        mode = 'L' if self.compressed_image.ndim == 2 else 'RGB'
        Image.fromarray(self.compressed_image, mode=mode).save(out_path)
        self.last_save_path = out_path
        return out_path

    def estimate_last_save_size(self):
        """Return the file size in bytes of the last saved reconstructed image."""
        if not self.last_save_path or not os.path.exists(self.last_save_path):
            raise RuntimeError("No reconstructed image file found. Save an image first.")
        return os.path.getsize(self.last_save_path)

    def get_compression_info(self):
        if not self.best_params:
            return None
        return {
            'levels': self.levels,
            'target_psnr': self.target_psnr,
            **self.best_params,
            **self.compression_stats
        }


def main():
    parser = argparse.ArgumentParser(
        description="Compress images using Haar wavelet transform with optional PSNR target or fixed threshold."
    )
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('-p', '--psnr', type=float, default=30,
                        help='Target PSNR value for automatic threshold selection (default: 30)')
    parser.add_argument('-t', '--threshold', type=float,
                        help='Fixed threshold for compression (overrides PSNR target)')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Directory to save the reconstructed image')
    args = parser.parse_args()

    compressor = HaarCompressor(output_dir=args.output_dir)
    try:
        if args.threshold is not None:
            compressor.compress_image_by_threshold(args.image_path, args.threshold)
        else:
            compressor.compress_image(args.image_path, target_psnr=args.psnr)

        out_path = compressor.save_reconstructed_image()
        size_bytes = compressor.estimate_last_save_size()
        info = compressor.get_compression_info()

        print(f"Reconstructed image saved to: {out_path}")
        print(f"File size: {size_bytes} bytes")
        print("Compression stats:")
        for k, v in info.items():
            print(f"  {k}: {v}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
