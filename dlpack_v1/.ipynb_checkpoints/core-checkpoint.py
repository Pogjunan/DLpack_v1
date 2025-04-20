## import ##
import re
import os
import imageio
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections.abc import Iterable
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
from dlpack_v1.dataset import ImageDataset

##function ##

def truncate(value, max_length=200):
    """Truncate string representation of a value if it exceeds max_length."""
    value_str = str(value)
    return value_str if len(value_str) <= max_length else value_str[:max_length] + "..."

def display_items(items, max_head=5, max_tail=5):
    """Display head and tail of items, skipping the middle if necessary."""
    if not hasattr(items, "__len__"):
        print(f"Type: {type(items).__name__}, Value: {truncate(items)}")
        return
    
    total_items = len(items)
    print(f"Total items: {total_items}")
    
    for idx, item in enumerate(items):
        if idx >= max_head and idx < total_items - max_tail:
            if idx == max_head:
                print("...")
            continue
        print(f"[{idx}] Type: {type(item).__name__}, Value: {truncate(item)}")

def show(item, max_depth=2, max_head=5, max_tail=5):
    """
    Display structured overview of an item.
    Supports dictionaries, lists, or any iterable.
    """
    if isinstance(item, dict):
        print(f"Dictionary Overview: {len(item)} keys")
        for idx, (key, value) in enumerate(item.items()):
            print(f"[{idx}] Key: {key}, Type: {type(value).__name__}, Value: {truncate(value)}")
    elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
        print("List Overview:")
        display_items(item, max_head=max_head, max_tail=max_tail)
    else:
        print(f"Unsupported Type: {type(item).__name__}, Value: {truncate(item)}")


import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    def step(self, closure=None):
        assert closure is not None, "SAM optimizer requires a closure function."
        closure = torch.enable_grad()(closure)

        # 첫 번째 순전파 및 역전파
        loss = closure()  # loss 계산 및 backward 수행

        # Step 1: 파라미터를 업데이트하여 경사면을 따라 이동
        self.first_step()

        # 두 번째 순전파 및 역전파 (그래디언트 계산 필요 없음)
        with torch.no_grad():
            closure()

        # Step 2: 원래 파라미터로 복원하고 실제 업데이트 수행
        self.second_step()

        return loss  # 첫 번째 loss 반환

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.abs(p) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # 원래 파라미터로 복원

        self.base_optimizer.step()  # 실제 업데이트 수행

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

def image_animation(image_dir, output_gif, duration=0.7):
    """
    Create an animated GIF from a series of images in a directory.

    Args:
        image_dir (str or Path): Directory containing the image files.
        output_gif (str): Filename for the output GIF.
        duration (float): Duration for each frame in the GIF (in seconds).
    """
    # image_dir = "image_dir"  # Directory containing PNG images
    # output_gif = "2nd_trial.gif"  # Output GIF filename
    # image_animation(image_dir, output_gif, duration=0.7)
    
    # Convert image_dir to Path object if it's a string
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob("*.png"), key=lambda x: x.name)
    if not image_files:
        print("No image files found in the specified directory.")
        return

    with imageio.get_writer(output_gif, mode="I", duration=duration) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"Animation saved as {output_gif}")



def parse_filename(filename):
    # 예: 2022-07-02_0330_4_predicted.png
    # 파일명 패턴에서 날짜/시간 부분을 추출
    # 정규표현식: YYYY-MM-DD_HHMM 형식 추출
    # 나의 경우 여기서 _4_ , _predicted 또는 _target 은 공통적으로 들어가므로 제거 가능
    pattern = r'(\d{4}-\d{2}-\d{2})_(\d{4})'
    match = re.search(pattern, filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        # time_str: '0330' -> hour=03, minute=30
        hour = int(time_str[:2])
        minute = int(time_str[2:])
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dt = dt.replace(hour=hour, minute=minute)
        return dt
    else:
        return None

def image_animation_7day(pred_dir, target_dir, output_gif, font_path=None):
    """
    7일 단위로 이미지를 끊어서 애니메이션을 만든다.
    한 덩어리는 20초 동안 표시, 덩어리 사이에 10초 쉬는 구간을 둔다.
    왼쪽에 predicted, 오른쪽에 target 이미지를 배치.
    각 이미지 하단에 해당 시간 표시,
    그리고 7일 덩어리 하단 중앙에 해당 기간 표시(예: 2022-07-01 ~ 2022-07-08)
    """

    pred_dir = Path(pred_dir)
    target_dir = Path(target_dir)

    pred_files = sorted(pred_dir.glob("*.png"), key=lambda x: x.name)
    target_files = sorted(target_dir.glob("*.png"), key=lambda x: x.name)

    pred_dict = {}
    for pf in pred_files:
        dt = parse_filename(pf.name)
        if dt:
            pred_dict[dt] = pf

    pairs = []
    for tf in target_files:
        dt = parse_filename(tf.name)
        if dt and dt in pred_dict:
            pairs.append((dt, pred_dict[dt], tf))

    pairs.sort(key=lambda x: x[0])
    #디버깅 확실히하기 (파일 못 찾는 경우.. 시간버리는 일 방지)
    if not pairs:
        print("No matched image pairs found. (파일위치나 파일명 바꿔주세요)")
        return

    # 7일 단위로 끊기
    # 첫 이미지 시간을 기준으로 7일 묶음 생성
    chunks = []
    chunk = []
    start_time = pairs[0][0]
    end_time = start_time + timedelta(days=7)
    
    for p in pairs:
        dt = p[0]
        if dt < end_time:
            # 해당 7일 범위에 포함
            chunk.append(p)
        else:
            # 새로운 7일 범위 시작
            if chunk:
                chunks.append(chunk)
            chunk = [p]
            start_time = dt
            end_time = start_time + timedelta(days=7)
    # 마지막 chunk 추가
    if chunk:
        chunks.append(chunk)

    # 폰트 설정
    # font_path는 사용자가 제공하거나 시스템 폰트를 지정한다.
    # font_path가 None일 경우 기본 PIL 폰트를 사용.
    if font_path and Path(font_path).exists():
        font = ImageFont.truetype(str(font_path), 20)
    else:
        font = ImageFont.load_default()


    with imageio.get_writer(output_gif, mode="I") as writer:
        for c in chunks:
            # c는 7일짜리 덩어리
            # 해당 chunk 의 표시 기간: 총 20초
            # 이미지 개수: len(c)
            # 각 프레임 duration: 20 / len(c)
            n = len(c)
            if n == 0:
                continue
            frame_duration = 20.0 / n
            
            start_dt = c[0][0]
            end_dt = c[-1][0]
            date_range_text = f"{start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}"

            for (dt, pred_file, target_file) in c:
                pred_img = Image.open(pred_file).convert("RGB")
                target_img = Image.open(target_file).convert("RGB")
                w1, h1 = pred_img.size
                w2, h2 = target_img.size
                out_w = w1 + w2
                out_h = max(h1, h2) + 50
                combined = Image.new("RGB", (out_w, out_h), (255, 255, 255))
                combined.paste(pred_img, (0, 0))
                combined.paste(target_img, (w1, 0))

                draw = ImageDraw.Draw(combined)
                # 날짜/시간 텍스트: 각 프레임 개별 이미지의 날짜 시간
                time_text = dt.strftime("%Y-%m-%d %H:%M")
                # text 위치 (가운데 정렬을 위해 텍스트 길이를 고려)
                # 날짜/시간 텍스트 크기 계산
                bbox = font.getbbox(time_text)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(((out_w - tw) / 2, out_h - th - 5), time_text, fill="black", font=font)
                
                bbox = font.getbbox(date_range_text)
                trw, trh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(((out_w - trw) / 2, out_h - th - trh - 30), date_range_text, fill="black", font=font)

                frame = combined.copy()
                # frame은 PIL Image 객체
                frame_array = np.array(frame)  
# PIL Image -> numpy array 변환 ( 파일 경로나 파일과 유사한 객체를 기대때문에 변경필요)
                #writer.append_data(frame_array, duration=frame_duration) <--문제코드
                #메타데이터(meta) 딕셔너리를 통해 전달가능 
                writer.append_data(frame_array, meta={'duration': frame_duration})
                #이후 버전 바뀌어서 문제 발생시 imageio.mimsave() -> mageio.mimsave('animation.gif', frames, duration=0.05) 사용

                
                
                pause_frame = Image.new("RGB", (out_w, out_h), (255, 255, 255))
                pause_array = np.array(pause_frame)
                writer.append_data(pause_array)  

    print(f"Animation saved as {output_gif}")




def image_animation_7day1(pred_dir, target_dir, output_gif, font_path=None):
    pred_dir = Path(pred_dir)
    target_dir = Path(target_dir)

    pred_files = sorted(pred_dir.glob("*.png"), key=lambda x: x.name)
    target_files = sorted(target_dir.glob("*.png"), key=lambda x: x.name)

    pred_dict = {}
    for pf in pred_files:
        dt = parse_filename(pf.name)
        if dt:
            pred_dict[dt] = pf

    pairs = []
    for tf in target_files:
        dt = parse_filename(tf.name)
        if dt and dt in pred_dict:
            pairs.append((dt, pred_dict[dt], tf))

    pairs.sort(key=lambda x: x[0])
    if not pairs:
        print("No matched image pairs found.")
        return

    chunks = []
    chunk = []
    start_time = pairs[0][0]
    end_time = start_time + timedelta(days=7)

    for p in pairs:
        dt = p[0]
        if dt < end_time:
            chunk.append(p)
        else:
            if chunk:
                chunks.append(chunk)
            chunk = [p]
            start_time = dt
            end_time = start_time + timedelta(days=7)
    if chunk:
        chunks.append(chunk)

    if font_path and Path(font_path).exists():
        font = ImageFont.truetype(str(font_path), 20)
    else:
        font = ImageFont.load_default()

    frames = []
    durations = []

    for c in chunks:
        n = len(c)
        if n == 0:
            continue
        frame_duration = 20.0 / n

        start_dt = c[0][0]
        end_dt = c[-1][0]
        date_range_text = f"{start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}"

        out_w, out_h = None, None

        for (dt, pred_file, target_file) in c:
            pred_img = Image.open(pred_file).convert("RGB")
            target_img = Image.open(target_file).convert("RGB")

            w1, h1 = pred_img.size
            w2, h2 = target_img.size

            if out_w is None:
                out_w = w1 + w2
                out_h = max(h1, h2) + 50

            combined = Image.new("RGB", (out_w, out_h), (255, 255, 255))
            combined.paste(pred_img, (0, 0))
            combined.paste(target_img, (w1, 0))

            draw = ImageDraw.Draw(combined)
            time_text = dt.strftime("%Y-%m-%d %H:%M")
            bbox = font.getbbox(time_text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((out_w - tw) / 2, out_h - th - 5), time_text, fill="black", font=font)

            bbox = font.getbbox(date_range_text)
            trw, trh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((out_w - trw) / 2, out_h - th - trh - 30), date_range_text, fill="black", font=font)

            frame_array = np.array(combined)
            frames.append(frame_array)
            durations.append(frame_duration)
        #10초 공백 프레임 삽입
        # 공백 프레임(10초)을 주기 위해, 프레임 하나를 1초 단위로 10번 넣는 방법 사용 (느려도 무방)
        pause_frame = Image.new("RGB", (out_w, out_h), (255, 255, 255))
        pause_array = np.array(pause_frame)
        repeat_count = 200  
        for _ in range(repeat_count):
            frames.append(pause_array)
            durations.append(1.0)  

    imageio.mimsave(output_gif, frames, duration=durations, loop=0)
    print(f"Animation saved as {output_gif}")



from pathlib import Path
import imageio
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta
import re
import numpy as np

def parse_filename(filename):
    pattern = r'(\d{4}-\d{2}-\d{2})_(\d{4})'
    match = re.search(pattern, filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        hour = int(time_str[:2])
        minute = int(time_str[2:])
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dt = dt.replace(hour=hour, minute=minute)
        return dt
    else:
        return None

def image_animation_7day2(pred_dir, target_dir, output_gif, font_path=None):
    pred_dir = Path(pred_dir)
    target_dir = Path(target_dir)
    output_gif = Path(output_gif)  # Path 객체로 변환

    pred_files = sorted(pred_dir.glob("*.png"), key=lambda x: x.name)
    target_files = sorted(target_dir.glob("*.png"), key=lambda x: x.name)

    pred_dict = {}
    for pf in pred_files:
        dt = parse_filename(pf.name)
        if dt:
            pred_dict[dt] = pf

    pairs = []
    for tf in target_files:
        dt = parse_filename(tf.name)
        if dt and dt in pred_dict:
            pairs.append((dt, pred_dict[dt], tf))

    pairs.sort(key=lambda x: x[0])
    if not pairs:
        print("No matched image pairs found.")
        return

    # 7일 단위로 끊기
    chunks = []
    chunk = []
    start_time = pairs[0][0]
    end_time = start_time + timedelta(days=7)

    for p in pairs:
        dt = p[0]
        if dt < end_time:
            chunk.append(p)
        else:
            if chunk:
                chunks.append(chunk)
            chunk = [p]
            start_time = dt
            end_time = start_time + timedelta(days=7)
    if chunk:
        chunks.append(chunk)

    # 폰트 설정
    if font_path and Path(font_path).exists():
        font = ImageFont.truetype(str(font_path), 20)
    else:
        font = ImageFont.load_default()

    # 각 chunk마다 별도의 GIF 파일 생성
    for i, c in enumerate(chunks, start=1):
        n = len(c)
        if n == 0:
            continue
        frame_duration = 20.0 / n

        start_dt = c[0][0]
        end_dt = c[-1][0]
        date_range_text = f"{start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}"

        out_w, out_h = None, None

        # 이 chunk에 대한 frames, durations 초기화
        frames = []
        durations = []

        for (dt, pred_file, target_file) in c:
            pred_img = Image.open(pred_file).convert("RGB")
            target_img = Image.open(target_file).convert("RGB")

            w1, h1 = pred_img.size
            w2, h2 = target_img.size

            if out_w is None:
                out_w = w1 + w2
                out_h = max(h1, h2) + 50

            combined = Image.new("RGB", (out_w, out_h), (255, 255, 255))
            combined.paste(pred_img, (0, 0))
            combined.paste(target_img, (w1, 0))

            draw = ImageDraw.Draw(combined)
            time_text = dt.strftime("%Y-%m-%d %H:%M")
            bbox = font.getbbox(time_text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((out_w - tw) / 2, out_h - th - 5), time_text, fill="black", font=font)

            bbox = font.getbbox(date_range_text)
            trw, trh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((out_w - trw) / 2, out_h - th - trh - 30), date_range_text, fill="black", font=font)

            frame_array = np.array(combined)
            frames.append(frame_array)
            durations.append(frame_duration)


        pause_frame = Image.new("RGB", (out_w, out_h), (255, 255, 255))
        pause_array = np.array(pause_frame)
        repeat_count = 200  # 5초 / 0.2초 = 25
        for _ in range(repeat_count):
            frames.append(pause_array)
            durations.append(1.0)

        # 현재 chunk를 별도 gif로 저장
        chunk_output_gif = output_gif.parent / f"{output_gif.stem}_{i}{output_gif.suffix}"
        imageio.mimsave(chunk_output_gif, frames, duration=durations, loop=0)
        print(f"Chunk {i} animation saved as {chunk_output_gif}")


import time
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from datetime import datetime, timedelta

def parse_filename(filename: str):
    """
    사용 중이신 시간 파싱 함수를 그대로 사용한다고 가정.
    예: '2022-07-01_0530_0_predicted.png' -> datetime 객체로 변환
    """
    # 아래는 단순 예시. 실제 파일 이름 패턴에 맞춰 수정해서 사용하세요.
    # '2022-07-01_0530' 형태만 파싱하고, 뒤의 '_0_predicted'는 무시한다고 가정
    try:
        base = filename.split('_predicted')[0]  # '2022-07-01_0530_0'
        # base.split('_') -> ['2022-07-01','0530','0']
        date_str = base.split('_')[0]           # '2022-07-01'
        time_str = base.split('_')[1]           # '0530'
        # 필요하다면 index = base.split('_')[2] 등을 쓰실 수도 있음.

        # '2022-07-01' + '0530' -> datetime 변환
        date_part = datetime.strptime(date_str, "%Y-%m-%d")
        hour = int(time_str[:2])
        minute = int(time_str[2:])
        dt = date_part.replace(hour=hour, minute=minute)
        return dt
    except:
        return None


from pathlib import Path
from datetime import timedelta
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio  # v2 인터페이스 사용
import numpy as np
import time  # 만약 디버깅용 딜레이가 필요하면 사용
from pathlib import Path
from datetime import timedelta
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import numpy as np

from pathlib import Path
from datetime import timedelta
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import numpy as np

def animation_image_newcode_mp4(pred_dirs, target_dir, output_mp4, font_path=None):
    """
    pred_dirs: 예측 이미지를 모아둔 디렉터리들의 리스트 (예: ["rs_pred_0", "rs_pred_1", ...])
    target_dir: 타겟 이미지를 모아둔 디렉터리 (예: "rs_target_0")
    output_mp4: 결과 MP4 동영상 저장 경로 (예: "result.mp4")
    font_path:  폰트 경로 (없으면 PIL Default Font)
    
    동작 개요:
      1. 타겟 디렉터리의 PNG 파일들에서 파일명으로부터 날짜/시간 정보를 파싱하여
         target_dict[dt] = 파일경로 로 구성합니다.
      2. 예측 디렉터리들에 대해, 동일 시간(dt)에 해당하는 예측 이미지들을 모아
         pred_dict[dt] = [파일경로_dir0, 파일경로_dir1, ...] 로 구성합니다.
      3. 타겟 시간들을 7일 단위로 분할하여 chunk를 구성합니다.
      4. 각 chunk 내에서 각 시각(dt)에 대해, 왼쪽에는 예측 이미지, 오른쪽에는 타겟 이미지를 붙이고,
         하단에 시간/날짜 텍스트를 그린 후, 추가 여백 영역에 예측 디렉터리 이름(예: "rs_pred_0")을 중앙 아래쪽에 표시한
         하나의 합성 프레임을 생성합니다.
      5. 각 chunk마다 생성된 프레임들을 모아 imageio.mimsave()를 통해 MP4 동영상을 생성합니다.
    
    전제:
      - 모든 예측(pred) 및 타겟(target) 이미지는 512×512 크기의 RGB 이미지 (shape=(512,512,3))임.
      - parse_filename(filename) 함수는 파일명에서 datetime 객체를 반환한다고 가정.
      - 합성 캔버스는 좌우 합성으로 가로 512+512=1024, 세로는 512에 기본 텍스트 여백 50픽셀을 더해 562 픽셀이며,
        여기에 추가로 extra_margin(예, 30픽셀)을 더하여 전체 캔버스 높이를 562+30=592로 사용합니다.
    """
    # 1. 경로 객체 변환
    pred_dirs = [Path(d) for d in pred_dirs]
    target_dir = Path(target_dir)
    output_mp4 = Path(output_mp4)

    # 2. 타겟 파일 수집: target_dict[dt] = target 파일 경로
    target_files = sorted(target_dir.glob("*.png"), key=lambda x: x.name)
    target_dict = {}
    for tf in target_files:
        dt = parse_filename(tf.name)
        if dt:
            target_dict[dt] = tf

    target_times = sorted(target_dict.keys())
    if not target_times:
        print("No target files found.")
        return

    # 3. 예측 파일 수집: pred_dict[dt] = [파일경로, ...]
    num_dirs = len(pred_dirs)
    pred_dict = {}
    for i, pdir in enumerate(pred_dirs):
        for pf in sorted(pdir.glob("*.png"), key=lambda x: x.name):
            dt = parse_filename(pf.name)
            if dt:
                if dt not in pred_dict:
                    pred_dict[dt] = [None] * num_dirs
                pred_dict[dt][i] = pf

    # 4. 7일 간격으로 타겟 시간 chunk 분할
    chunks = []
    chunk = []
    start_time = target_times[0]
    end_time = start_time + timedelta(days=7)
    for dt in target_times:
        if dt < end_time:
            chunk.append(dt)
        else:
            if chunk:
                chunks.append(chunk)
            chunk = [dt]
            start_time = dt
            end_time = dt + timedelta(days=7)
    if chunk:
        chunks.append(chunk)

    # 5. 폰트 설정
    if font_path and Path(font_path).exists():
        font = ImageFont.truetype(str(font_path), 20)
    else:
        font = ImageFont.load_default()

    extra_margin = 30  # 예측 디렉터리 이름을 넣을 추가 여백

    # 6. 각 chunk별로 프레임 생성 및 MP4 저장 (fps=1)
    for idx, c_times in enumerate(chunks, start=1):
        if not c_times:
            continue

        frames = []
        start_dt = c_times[0]
        end_dt = c_times[-1]
        date_range_text = f"{start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}"
        out_w, base_h = None, None

        for dt in c_times:
            if dt not in target_dict:
                continue
            target_file = target_dict[dt]
            target_img = Image.open(target_file).convert("RGB")

            if dt not in pred_dict:
                continue
            pred_files_for_dt = pred_dict[dt]

            for pred_file in pred_files_for_dt:
                if pred_file is None:
                    continue

                pred_img = Image.open(pred_file).convert("RGB")
                w_pred, h_pred = pred_img.size  # 보통 (512,512)
                w_tgt, h_tgt = target_img.size   # (512,512)

                if out_w is None:
                    out_w = w_pred + w_tgt   # 512+512 = 1024
                    base_h = max(h_pred, h_tgt) + 50  # 512 + 50 = 562

                # 합성 캔버스 생성 (기존 합성 이미지)
                combined = Image.new("RGB", (out_w, base_h), (255, 255, 255))
                combined.paste(pred_img, (0, 0))
                combined.paste(target_img, (w_pred, 0))

                # 텍스트 추가 (시간, 날짜 범위) - 하단 중앙에 그림
                draw = ImageDraw.Draw(combined)
                time_text = dt.strftime("%Y-%m-%d %H:%M")
                bbox = font.getbbox(time_text)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(((out_w - tw) / 2, base_h - th - 5),
                          time_text, fill="black", font=font)
                bbox2 = font.getbbox(date_range_text)
                trw, trh = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
                draw.text(((out_w - trw) / 2, base_h - th - trh - 30),
                          date_range_text, fill="black", font=font)

                # 새 캔버스 생성: 기존 합성 이미지 위에 extra_margin 만큼의 여백 추가
                new_h = base_h + extra_margin  # 총 높이 562 + 30 = 592
                new_canvas = Image.new("RGB", (out_w, new_h), (255, 255, 255))
                new_canvas.paste(combined, (0, 0))
                draw_new = ImageDraw.Draw(new_canvas)
                # 예측 디렉터리 이름 (pred_source) 가져오기
                pred_source = pred_file.parent.name
                bbox_src = font.getbbox(pred_source)
                sw, sh = bbox_src[2] - bbox_src[0], bbox_src[3] - bbox_src[1]
                x_src = (out_w - sw) // 2
                y_src = base_h + (extra_margin - sh) // 2
                draw_new.text((x_src, y_src), pred_source, fill="black", font=font)

                # 최종 프레임 배열
                frame_array = np.array(new_canvas)
                frame_array = np.ascontiguousarray(frame_array)
                if frame_array.shape != (new_h, out_w, 3):
                    continue

                frames.append(frame_array)

        if not frames:
            print(f"No valid frames in chunk {idx}. Skipping MP4 generation.")
            continue

        chunk_output_mp4 = output_mp4.parent / f"{output_mp4.stem}_{idx}.mp4"
        try:
            imageio.mimsave(str(chunk_output_mp4), frames, fps=1, codec='libx264')
            print(f"Chunk {idx} MP4 saved as {chunk_output_mp4}")
        except Exception as e:
            print(f"Error saving chunk {idx} MP4: {e}")

# =============================================================================
# 사용 예시:
# =============================================================================
# pred_dirs = ["rs_pred_0", "rs_pred_1", "rs_pred_2", "rs_pred_3", "rs_pred_4", "rs_pred_5"]
# target_dir = "rs_target_0"
# output_mp4 = "lp_0_animation.mp4"
# font_path = "arial.ttf"  (옵션)
#
# 예시 호출:
# animation_image_newcode_mp4(pred_dirs, target_dir, output_mp4, font_path)



import numpy as np
from pathlib import Path
from datetime import timedelta
import imageio
from PIL import Image, ImageDraw, ImageFont

# parse_filename 함수는 파일명에서 datetime 객체를 반환한다고 가정합니다.
# 예: def parse_filename(fname): ... return datetime_object
import numpy as np
from pathlib import Path
from datetime import timedelta
import imageio
from PIL import Image, ImageDraw, ImageFont

# parse_filename 함수는 파일명에서 datetime 객체를 반환한다고 가정합니다.
# 예: def parse_filename(fname): ... return datetime_object

import numpy as np
import imageio
from pathlib import Path
from datetime import timedelta, datetime
from PIL import Image, ImageDraw, ImageFont
import re

def parse_filename(fname):
    """
    파일명에서 'YYYY-MM-DD_HHMM' 패턴을 추출하여 datetime 객체를 반환합니다.
    예: "2022-07-01_0500_1_predicted.png" -> datetime(2022, 7, 1, 5, 0)
    """
    pattern = r'(\d{4}-\d{2}-\d{2})_(\d{4})'
    match = re.search(pattern, fname)
    if match:
        date_str, time_str = match.groups()
        try:
            dt = datetime.strptime(date_str + time_str, '%Y-%m-%d%H%M')
            return dt
        except Exception as e:
            print(f"Error parsing datetime from {fname}: {e}")
    return None

def animation_image_newcode_mp4_new(pred_dirs, target_dir, output_mp4, font_path=None):
    """
    [수정된 알고리즘]
    - target_dir에서 30분 간격으로 촬영된 이미지들을 시간순으로 정렬합니다.
    - 슬라이딩 윈도우로 연속된 3개의 target 이미지를 선택하고,
      각 컬럼의 시간에 맞춰 pred_dirs의 해당 예측 이미지를 가져옵니다.
        * 첫 번째 컬럼: target 시간 t₀ (예측: 30분 후, pred_dirs[0])
        * 두 번째 컬럼: target 시간 t₁ (예측: 60분 후, pred_dirs[1])
        * 세 번째 컬럼: target 시간 t₂ (예측: 90분 후, pred_dirs[2])
    - 한 프레임은 2행 3열 매트릭스 형태로 구성되며,
         상단 행: 예측 이미지  
         하단 행: target 이미지  
      각 셀 하단에는 해당 날짜/시간 텍스트를 표시하는데, 기존 위치보다 30픽셀 아래에 그립니다.
    - 추가로, 상단 예측 이미지 영역 바로 위(사진 바깥 위, 헤더 영역)에 각 컬럼별로
         "After 30 minutes", "After 60 minutes", "After 90 minutes"
      텍스트를 중앙 정렬하여 표시합니다.
    - 30분씩 기준시간을 슬라이딩하며, 1초마다 프레임이 전환되는 MP4 동영상으로 저장합니다.
    """
    # 1. 경로 객체 변환
    pred_dirs = [Path(d) for d in pred_dirs]
    target_dir = Path(target_dir)
    output_mp4 = Path(output_mp4)
    
    # 2. target 파일 수집: target_dict[datetime] = 파일경로
    target_files = sorted(target_dir.glob("*.png"), key=lambda x: x.name)
    target_dict = {}
    for tf in target_files:
        dt = parse_filename(tf.name)
        if dt:
            target_dict[dt] = tf
    if not target_dict:
        print("No target files found.")
        return
    
    # 3. 각 예측 디렉터리에 대해 시간별 파일 딕셔너리 생성
    pred_dicts = {}
    for pd in pred_dirs:
        files = sorted(pd.glob("*.png"), key=lambda x: x.name)
        temp_dict = {}
        for f in files:
            dt = parse_filename(f.name)
            if dt:
                temp_dict[dt] = f
        pred_dicts[pd] = temp_dict

    # 4. target 시간 정렬 (30분 간격이라 가정)
    sorted_target_times = sorted(target_dict.keys())
    
    # 슬라이딩 윈도우: 3개 연속 이미지 사용
    frames = []
    cell_w, cell_h = 512, 512       # 각 셀의 이미지 크기
    base_text_margin = 30           # 기존 텍스트 영역 높이
    extra_time_offset = 30          # 텍스트를 추가로 아래로 이동할 픽셀
    new_text_margin = base_text_margin + extra_time_offset  # 새 텍스트 영역 높이 (60)
    header_margin = 30              # 예측 이미지 헤더 텍스트 영역 높이 (상단)
    
    # composite 전체 크기 계산:
    # width = 3 * cell_w
    # height = header_margin + (상단 행: cell_h + new_text_margin) + (하단 행: cell_h + new_text_margin)
    comp_width = 3 * cell_w
    comp_height = header_margin + (cell_h + new_text_margin) + (cell_h + new_text_margin)
    
    for i in range(len(sorted_target_times) - 2):
        # 슬라이딩 윈도우에서 target 시간 3개 선택
        t0 = sorted_target_times[i]
        t1 = sorted_target_times[i+1]
        t2 = sorted_target_times[i+2]
        
        # 각 컬럼에 대해 예측 이미지 존재 확인
        if (t0 not in pred_dicts[pred_dirs[0]] or
            t1 not in pred_dicts[pred_dirs[1]] or
            t2 not in pred_dicts[pred_dirs[2]]):
            print(f"Skipping window starting at {t0.strftime('%Y-%m-%d %H:%M')} because one or more predicted images are missing.")
            continue
        
        try:
            # target 이미지 로드 (하단 행)
            target_img0 = Image.open(target_dict[t0]).convert("RGB")
            target_img1 = Image.open(target_dict[t1]).convert("RGB")
            target_img2 = Image.open(target_dict[t2]).convert("RGB")
            # 예측 이미지 로드 (상단 행)
            pred_img0 = Image.open(pred_dicts[pred_dirs[0]][t0]).convert("RGB")
            pred_img1 = Image.open(pred_dicts[pred_dirs[1]][t1]).convert("RGB")
            pred_img2 = Image.open(pred_dicts[pred_dirs[2]][t2]).convert("RGB")
        except Exception as e:
            print(f"Error loading images for window starting at {t0.strftime('%Y-%m-%d %H:%M')}: {e}")
            continue
        
        # 5. composite 캔버스 생성 및 드로잉 객체 생성
        composite = Image.new("RGB", (comp_width, comp_height), (255, 255, 255))
        draw = ImageDraw.Draw(composite)
        
        # 폰트 로드 (옵션)
        if font_path and Path(font_path).exists():
            font = ImageFont.truetype(str(font_path), 20)
        else:
            font = ImageFont.load_default()
        
        # ───────────────────────────────
        # (A) 헤더 텍스트 추가 (예측 이미지 위, header 영역)
        def draw_header_text(cell_idx, text):
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = cell_idx * cell_w + (cell_w - text_w) // 2
            y = (header_margin - text_h) // 2  # header 영역 중앙 배치
            draw.text((x, y), text, fill="black", font=font)
        
        draw_header_text(0, "After 30 minutes")
        draw_header_text(1, "After 60 minutes")
        draw_header_text(2, "After 90 minutes")
        # ───────────────────────────────
        
        # ───────────────────────────────
        # (B) 이미지 배치
        # 상단 행 (예측 이미지): header_margin부터 시작
        composite.paste(pred_img0, (0, header_margin))
        composite.paste(pred_img1, (cell_w, header_margin))
        composite.paste(pred_img2, (2 * cell_w, header_margin))
        
        # 하단 행 (target 이미지): 상단 행 영역 다음에 배치
        bottom_row_y = header_margin + cell_h + new_text_margin
        composite.paste(target_img0, (0, bottom_row_y))
        composite.paste(target_img1, (cell_w, bottom_row_y))
        composite.paste(target_img2, (2 * cell_w, bottom_row_y))
        # ───────────────────────────────
        
        # ───────────────────────────────
        # (C) 시간 텍스트 추가 (각 셀 하단, 기존 위치보다 30픽셀 아래에)
        def draw_text(cell_idx, text, row):
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = cell_idx * cell_w + (cell_w - text_w) // 2
            if row == 0:
                # 상단 행 (예측 이미지)의 경우:
                # 이미지는 header_margin ~ header_margin+cell_h 이며, 기존 텍스트 위치: header_margin + cell_h - text_h
                base_y = header_margin + cell_h
            else:
                # 하단 행 (target 이미지)의 경우:
                # 이미지는 bottom_row_y ~ bottom_row_y+cell_h 이며, 기존 텍스트 위치: bottom_row_y + cell_h - text_h
                base_y = bottom_row_y + cell_h
            y = base_y - text_h + extra_time_offset  # 기존 위치보다 30픽셀 아래로
            draw.text((x, y), text, fill="black", font=font)
        
        # 상단(예측) 행의 시간 텍스트
        draw_text(0, t0.strftime("%Y-%m-%d %H:%M"), 0)
        draw_text(1, t1.strftime("%Y-%m-%d %H:%M"), 0)
        draw_text(2, t2.strftime("%Y-%m-%d %H:%M"), 0)
        # 하단(target) 행의 시간 텍스트
        draw_text(0, t0.strftime("%Y-%m-%d %H:%M"), 1)
        draw_text(1, t1.strftime("%Y-%m-%d %H:%M"), 1)
        draw_text(2, t2.strftime("%Y-%m-%d %H:%M"), 1)
        # ───────────────────────────────
        
        frames.append(np.array(composite))
    
    if not frames:
        print("No composite frames generated.")
        return
    from fractions import Fraction


    # 7. MP4 동영상으로 저장 (fps=1, 1초마다 프레임 전환)
    try:
        fps = Fraction(4, 3)  # 4/3 ≈ 1.33
        imageio.mimsave(str(output_mp4), frames, fps=fps, codec='libx264')
        print(f"MP4 video saved as {output_mp4}")
    except Exception as e:
        print(f"Error saving MP4: {e}")

# =============================================================================
# 사용 예시:
# =============================================================================
# pred_dirs = ["rs_pred_0_base", "rs_pred_1", "rs_pred_2"]
# target_dir = "rs_target_0"
# output_mp4 = "result.mp4"
# font_path = "arial.ttf"  (옵션)



import numpy as np

def LU(A):
    n, _ = A.shape
    L = np.identity(n)
    U = A.astype(float)
    
    print("초기 L:\n", L)
    print("초기 U:\n", U)
    
    for r in range(n-1):
        print(f"\n--------- r = {r} ---------")
        
        # L[r:,r] 업데이트
        L[r+1:,r] = U[r+1:,r] / U[r,r]
        print(f"L[r+1:,r] = U[r+1:,r] / U[r,r]")
        print(f"L[{r+1}:,{r}] = U[{r+1}:,{r}] / U[{r},{r}] = {U[r+1:,r]} / {U[r,r]}")
        print("업데이트된 L:\n", L)
        
        # U 업데이트
        for i in range(r+1,n):
            print(f"\ni = {i}일 때")
            print(f"U[{i},{r}:] = U[{i},{r}:] - (U[{i},{r}]/U[{r},{r}])*U[{r},{r}:]")
            print(f"U[{i},{r}:] = {U[i,r:]} - ({U[i,r]}/{U[r,r]})*{U[r,r:]}")
            
            multiplier = U[i,r]/U[r,r]
            print(f"곱셈 계수: {multiplier}")
            
            U[i,r:] = U[i,r:] - multiplier * U[r,r:]
            print(f"업데이트된 U[{i},{r}:] = {U[i,r:]}")
        
        print("\n현재 U:\n", U)
    
    print("\n최종 L:\n", L)
    print("최종 U:\n", U)
    return L, U

def LU1(A):
    print("    n, _ = A.shape")
    print(" L = np.identity(n))")
    print(" U = A.astype(float)")

def LU2(A):
    print("for r in range(n-1):")
    print("    L[r+1:,r] = U[r+1:,r] / U[r,r]")

def LU3(A):
    print("for i in range(r+1,n):")
    print("U[i,r:] = U[i,r:] -  U[i,r]/U[r,r] *U[r,r:]") 


def LbSolver1(L,b):
    print("    n = len(L)")
    print(" y = np.zeros(n)")
    print("y[0] = b[0]")

def LbSolver2(L,b):
    print("        for i in range(1,n):")
    print(" y[i] = b[i] - L[i,:i] @ y[:i]")
    print("    return y")

    
def LbSolver(L, b):
    print("\n===== LbSolver 실행 =====")
    print("입력 L 행렬:\n", L)
    print("입력 b 벡터:", b)
    
    n = len(L)
    y = np.zeros(n)
    
    print("\n첫 번째 방정식 계산:")
    print(f"y[0] = b[0] = {b[0]}")
    y[0] = b[0]
    print(f"y = {y}")
    
    for i in range(1, n):
        print(f"\n{i+1}번째 방정식 계산:")
        print(f"L[{i},:i] = {L[i,:i]}")
        print(f"y[:i] = {y[:i]}")
        dot_product = L[i,:i] @ y[:i]
        print(f"L[{i},:i] @ y[:i] = {dot_product}")
        
        y[i] = b[i] - dot_product
        print(f"y[{i}] = b[{i}] - dot_product = {b[i]} - {dot_product} = {y[i]}")
        print(f"현재 y = {y}")
    
    print("\n===== LbSolver 결과 =====")
    print("최종 y =", y)
    return y



def UbSolver1(U,b):
    
    
    print("    n = len(U)")
    print(" x = np.zeros(n)")

def UbSolver2(U,b):
    print("x[n-1] = b[n-1] / U[n-1, n-1]")
    print("for i in range(n-2, -1, -1):")
    print("x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]")
        
    
def UbSolver(U, b):
    print("\n===== UbSolver 실행 =====")
    print("입력 U 행렬:\n", U)
    print("입력 b 벡터:", b)
    
    n = len(U)
    x = np.zeros(n)
    
    print("\n마지막 방정식부터 계산 시작:")
    print(f"x[{n-1}] = b[{n-1}] / U[{n-1},{n-1}] = {b[n-1]} / {U[n-1,n-1]} = {b[n-1]/U[n-1,n-1]}")
    x[n-1] = b[n-1] / U[n-1, n-1]
    print(f"x = {x}")
    
    for i in range(n-2, -1, -1):
        print(f"\n{i+1}번째 방정식 계산 (역순):")
        print(f"U[{i},i+1:] = {U[i,i+1:]}")
        print(f"x[i+1:] = {x[i+1:]}")
        
        dot_product = np.dot(U[i, i+1:], x[i+1:])
        print(f"U[{i},i+1:] @ x[i+1:] = {dot_product}")
        
        x[i] = (b[i] - dot_product) / U[i, i]
        print(f"x[{i}] = (b[{i}] - dot_product) / U[{i},{i}] = ({b[i]} - {dot_product}) / {U[i,i]} = {x[i]}")
        print(f"현재 x = {x}")
    
    print("\n===== UbSolver 결과 =====")
    print("최종 x =", x)
    return x