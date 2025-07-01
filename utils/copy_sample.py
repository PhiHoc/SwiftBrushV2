import os
import shutil
import pandas as pd

def copy_images_and_metadata(source_dir: str, target_dir: str, metadata_filename: str = 'meta.csv'):
    """
    Sao chép các tệp hình ảnh và tệp CSV siêu dữ liệu từ thư mục nguồn sang thư mục đích.

    Hàm này đọc một tệp CSV siêu dữ liệu để tìm đường dẫn của các tệp hình ảnh,
    sao chép các hình ảnh đó vào một thư mục đích trong khi vẫn giữ nguyên cấu trúc thư mục con,
    và cuối cùng sao chép tệp siêu dữ liệu vào thư mục đích.

    Args:
        source_dir (str): Đường dẫn đến thư mục nguồn chứa tệp meta.csv và các thư mục con chứa ảnh.
        target_dir (str): Đường dẫn đến thư mục đích nơi dữ liệu sẽ được sao chép vào.
        metadata_filename (str): Tên của tệp CSV siêu dữ liệu. Mặc định là 'meta.csv'.
    """
    # Bước 1: Kiểm tra xem thư mục nguồn có tồn tại không
    if not os.path.isdir(source_dir):
        print(f"Lỗi: Thư mục nguồn '{source_dir}' không tồn tại.")
        return

    # Tạo đường dẫn đầy đủ đến tệp metadata
    metadata_path = os.path.join(source_dir, metadata_filename)

    # Kiểm tra xem tệp metadata có tồn tại không
    if not os.path.isfile(metadata_path):
        print(f"Lỗi: Tệp siêu dữ liệu '{metadata_path}' không được tìm thấy.")
        return

    # Bước 2: Tạo thư mục đích nếu nó chưa tồn tại
    os.makedirs(target_dir, exist_ok=True)
    print(f"Đã tạo hoặc xác nhận thư mục đích: '{target_dir}'")

    try:
        # Bước 3: Đọc tệp CSV bằng pandas
        # Giả sử cột chứa đường dẫn hình ảnh có tên là 'Link' dựa trên hình ảnh của bạn
        df = pd.read_csv(metadata_path)
        image_path_column = 'Link' # <-- THAY ĐỔI TÊN CỘT NÀY NẾU CẦN

        if image_path_column not in df.columns:
            print(f"Lỗi: Không tìm thấy cột '{image_path_column}' trong tệp '{metadata_filename}'.")
            print(f"Các cột có sẵn: {df.columns.tolist()}")
            return

        # Bước 4: Lặp qua từng đường dẫn ảnh và sao chép tệp
        print("Bắt đầu sao chép hình ảnh...")
        for rel_path in df[image_path_column]:
            # Tạo đường dẫn nguồn và đích đầy đủ cho mỗi hình ảnh
            full_source_path = os.path.join(source_dir, rel_path)
            full_target_path = os.path.join(target_dir, rel_path)

            # Tạo thư mục con trong thư mục đích nếu cần
            target_image_dir = os.path.dirname(full_target_path)
            os.makedirs(target_image_dir, exist_ok=True)

            # Sao chép tệp hình ảnh
            if os.path.exists(full_source_path):
                shutil.copy2(full_source_path, full_target_path)
            else:
                print(f"Cảnh báo: Tệp nguồn '{full_source_path}' không tồn tại và sẽ được bỏ qua.")

        print("Đã sao chép thành công tất cả hình ảnh.")

        # Bước 5: Sao chép tệp metadata.csv vào thư mục đích
        shutil.copy2(metadata_path, os.path.join(target_dir, metadata_filename))
        print(f"Đã sao chép thành công tệp siêu dữ liệu '{metadata_filename}'.")

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

# --- Ví dụ cách sử dụng ---
if __name__ == '__main__':
    # Thiết lập thư mục nguồn và đích của bạn tại đây
    # Ví dụ:
    source_directory = '/path/to/your/source/folder'
    target_directory = '/path/to/your/target/folder'
    copy_images_and_metadata(source_directory, target_directory)

    # Dọn dẹp
    # shutil.rmtree(source_directory)
    # shutil.rmtree(target_directory)