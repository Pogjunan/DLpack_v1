import pandas as pd
import numpy as np
class ImageDataset:
    def __init__(self, csv_path="./train.csv"):
        self.df = pd.read_csv(csv_path)
        self.base_dir = os.path.join("..", "seung")
        
        self.df["file_path"] = self.df["file_name"].apply(lambda x: os.path.join(self.base_dir, x))

        def get_image_size(file_path):
            try:
                with Image.open(file_path) as img:
                    return img.size  # (width, height)
            except Exception:
                print(f"이미지 로드 중 오류 발생: {file_path} -> {Exception}")
                return None  #평범
        
        self.df["image_size"] = self.df["file_path"].apply(get_image_size)
        self.image_sizes = self.df["image_size"].dropna().tolist()  # None 값 제외
        unique_sizes = set(self.image_sizes) #unique 한지 판단 가능
        if len(unique_sizes) > 1:
            print("경고: 이미지들의 크기가 동일하지 않다.", unique_sizes)
        else:
            common_size = unique_sizes.pop() if unique_sizes else None
            print("모든 이미지의 크기가 동일합니다:", common_size)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        """
        index (int): Dataset의 순서에 따른 인덱스 (0부터 시작)
        img_array (np.ndarray): 이미지의 픽셀값 배열, shape은 (channel, H, W)
        """
        row = self.df.iloc[index]
        img_path = row["file_path"]  # 미리 생성된 file_path 사용 (더 빠름)
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img_array = np.array(img)
        except Exception as e:
            raise RuntimeError(f"이미지 파일을 열 수 없습니다: {img_path}") from e

        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array

    def show_image(self, index):
        """
        dataset[index]의 numpy 배열을 다시 PIL 이미지로 변환하여 열기
        """
        img_array = self[index]  # (C, H, W)
        img_array = np.transpose(img_array, (1, 2, 0))  # (H, W, C)로 변환
        img = Image.fromarray(img_array)  # numpy 배열을 PIL 이미지로 변환
        img.show()
        