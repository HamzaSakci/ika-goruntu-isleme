import os
import shutil
import glob

def filter_duba_dataset(src_dir, dest_dir, duba_class_id=12):
    """
    Karışık YOLO veri setinden sadece 'duba' içeren resimleri ve etiketleri ayıklar.
    Duba sınıfı ID'si (data.yaml'a göre) 12'dir.
    Yeni veri setinde Duba'nın class_id'sini 0 olarak değiştirir.
    """
    subdirs = ['train', 'valid', 'test']
    
    for subdir in subdirs:
        src_images_dir = os.path.join(src_dir, subdir, 'images')
        src_labels_dir = os.path.join(src_dir, subdir, 'labels')
        
        dest_images_dir = os.path.join(dest_dir, subdir, 'images')
        dest_labels_dir = os.path.join(dest_dir, subdir, 'labels')
        
        if not os.path.exists(src_labels_dir):
            continue
            
        os.makedirs(dest_images_dir, exist_ok=True)
        os.makedirs(dest_labels_dir, exist_ok=True)
        
        label_files = glob.glob(os.path.join(src_labels_dir, "*.txt"))
        duba_count = 0
        
        for label_file in label_files:
            has_duba = False
            new_lines = []
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0 and int(parts[0]) == duba_class_id:
                        has_duba = True
                        # Sınıf ID'sini 0 yapıyoruz (sadece duba olduğu için)
                        parts[0] = '0'
                        new_lines.append(" ".join(parts) + "\n")
            
            if has_duba:
                # 1. Yeni etiketi yaz
                base_name = os.path.basename(label_file)
                dest_label_path = os.path.join(dest_labels_dir, base_name)
                with open(dest_label_path, 'w') as f:
                    f.writelines(new_lines)
                
                # 2. İlgili resmi kopyala (.jpg, .jpeg veya .png)
                img_name_no_ext = os.path.splitext(base_name)[0]
                possible_exts = ['.jpg', '.jpeg', '.png', '.JPG']
                
                for ext in possible_exts:
                    src_img_path = os.path.join(src_images_dir, img_name_no_ext + ext)
                    if os.path.exists(src_img_path):
                        dest_img_path = os.path.join(dest_images_dir, img_name_no_ext + ext)
                        shutil.copy2(src_img_path, dest_img_path)
                        break
                        
                duba_count += 1
                
        print(f"{subdir} klasöründen {duba_count} adet duba resmi ayıklandı.")

if __name__ == "__main__":
    src_dataset = r"C:\Users\hamza\OneDrive\Masaüstü\Ika\Denemeler_Montagne.v6-a.yolov8"
    dest_dataset = r"C:\Users\hamza\OneDrive\Masaüstü\Ika\ika-goruntu-isleme\data\duba_dataset"
    
    print("Veri seti ayıklanıyor...")
    filter_duba_dataset(src_dataset, dest_dataset, duba_class_id=12)
    print("İşlem tamamlandı! Veriler 'data/duba_dataset' klasörüne kaydedildi.")
