import streamlit as st
from PIL import Image
from haar_matrix import HaarCompressor
import io

def main():
    st.set_page_config(page_title="Demo Ảnh Nén", layout="centered")
    st.title("🔧 Chuyển ảnh thường thành ảnh nén bằng phương pháp Haar")
    st.markdown("Upload ảnh rồi chọn mức độ nén (threshold).")

    uploaded_file = st.file_uploader("📁 Chọn ảnh", type=["png", "jpg", "jpeg"])
    threshold = st.slider("Chọn threshold", min_value=1, max_value=500, value=100)

    if uploaded_file is not None:
        # Đọc ảnh gốc
        image = Image.open(uploaded_file)
        image_format = image.format.upper()  # "PNG", "JPEG", etc.
        file_extension = image_format.lower() if image_format != "JPEG" else "jpg"
        mime_type = f"image/{file_extension}"

        # Convert PIL image to temporary path (HaarCompressor expects path)
        image_path = f"temp_input.{file_extension}"
        image.save(image_path)

        # Khởi tạo và nén ảnh
        haar_compressor = HaarCompressor()
        haar_compressor.compress_image_by_threshold(image_path, threshold)

        # Lấy ảnh nén dưới dạng PIL.Image
        compressed_pil = Image.fromarray(haar_compressor.compressed_image)

        # Lưu ảnh nén vào buffer
        compressed_io = io.BytesIO()
        compressed_pil.save(compressed_io, format=image_format)
        compressed_io.seek(0)

        # Hiển thị ảnh
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Ảnh gốc", use_container_width=True)
        with col2:
            st.image(compressed_pil, caption=f"Ảnh nén (threshold={threshold})", use_container_width=True)

        # Nút tải ảnh nén
        st.download_button(
            label="📥 Tải ảnh nén",
            data=compressed_io,
            file_name=f"compressed.{file_extension}",
            mime=mime_type
        )

if __name__ == "__main__":
    main()
