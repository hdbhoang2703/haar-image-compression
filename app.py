import streamlit as st
from PIL import Image
from haar_matrix import HaarCompressor

def main():
    st.set_page_config(page_title="Demo Ảnh Nén", layout="centered")
    st.title("🔧 Chuyển ảnh thường thành ảnh nén bằng phương pháp Haar")
    st.markdown("Upload ảnh rồi chọn mức độ nén (threshold).")

    uploaded_file = st.file_uploader("📁 Chọn ảnh", type=["png", "jpg", "jpeg"])
    threshold = st.slider("Chọn threshold", min_value=1, max_value=100, value=30)

    if uploaded_file is not None:
        # Đọc ảnh gốc
        image = Image.open(uploaded_file)

        # Khởi tạo và nén ảnh
        haar_compressor = HaarCompressor(uploaded_file, threshold=threshold)
        haar_compressor.compress()
        compressed_io = haar_compressor.get_compressed_image_bytes()
        compressed_image = Image.open(compressed_io)

        # Hiển thị ảnh gốc và ảnh nén cạnh nhau
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Ảnh gốc", use_container_width=True)

        with col2:
            st.image(compressed_image, caption=f"Ảnh nén (threshold={threshold})", use_container_width=True)

        # Nút tải ảnh nén
        st.download_button(
            label="📥 Tải ảnh nén",
            data=compressed_io,
            file_name="compressed.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
