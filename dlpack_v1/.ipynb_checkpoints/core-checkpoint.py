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

def animation_image_newcode(pred_dirs, target_dir, output_mp4, font_path=None):
    """
    pred_dirs: 예측 이미지를 모아둔 디렉터리들의 리스트
               예: ["rs_pred_0", "rs_pred_1", ...]
    target_dir: 타겟 이미지를 모아둔 디렉터리 (예: "rs_target_0")
    output_mp4: 결과 MP4 동영상 저장 경로 (예: "result.mp4")
    font_path:  폰트 경로 (없으면 PIL Default Font)
    
    1. 타겟 디렉터리 내의 PNG 파일에서 파일명으로부터 날짜/시간을 파싱하여 target_dict[dt] = file_path를 만듭니다.
    2. 예측 디렉터리들에 대해, 동일 시간(dt)에 해당하는 예측 이미지들을 모아 pred_dict[dt] = [file_dir0, file_dir1, ...] 형태로 만듭니다.
    3. 타겟 시간들을 7일 단위로 chunk로 나눕니다.
    4. 각 chunk 내에서, 각 dt마다 타겟 이미지와 그에 해당하는 예측 이미지들을 불러와 좌우로 합성(composite)한 후,
       시간 텍스트와 날짜 범위 텍스트를 그려 넣고, 해당 프레임을 동영상의 한 프레임으로 추가합니다.
    5. 각 chunk마다 별도의 MP4 파일을 생성합니다.
    """
    # 1. Path 변환
    pred_dirs = [Path(d) for d in pred_dirs]
    target_dir = Path(target_dir)
    output_mp4 = Path(output_mp4)  # 기본 출력 파일명(확장자 포함)

    # 2. 타겟 파일 수집: target_dict[dt] = target_file
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

    # 3. 예측 파일 수집: pred_dict[dt] = [file_from_dir0, file_from_dir1, ...]
    num_dirs = len(pred_dirs)
    pred_dict = {}  # 각 시간 dt에 대해, 길이가 num_dirs인 리스트(없으면 None)
    for i, pdir in enumerate(pred_dirs):
        files_in_dir = sorted(pdir.glob("*.png"), key=lambda x: x.name)
        for pf in files_in_dir:
            dt = parse_filename(pf.name)
            if dt:
                if dt not in pred_dict:
                    pred_dict[dt] = [None] * num_dirs
                pred_dict[dt][i] = pf

    # 4. 7일 간격으로 타겟 시간 chunk로 나누기
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

    # 6. 각 chunk별로 MP4 동영상 생성
    # 원본 output_mp4의 이름을 기준으로, chunk별로 _1, _2, ... 접미사를 붙임.
    for idx, c_times in enumerate(chunks, start=1):
        if not c_times:
            continue

        # 동영상 파일 이름 구성
        chunk_output_mp4 = output_mp4.parent / f"{output_mp4.stem}_{idx}.mp4"

        # MP4 writer 생성, fps=1 (즉, 각 프레임이 1초 동안 표시)
        writer = imageio.get_writer(str(chunk_output_mp4), fps=1, codec='libx264')
        
        # 날짜 범위 텍스트 (예: "2022-07-01 ~ 2022-07-07")
        start_dt = c_times[0]
        end_dt = c_times[-1]
        date_range_text = f"{start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}"
        
        out_w, out_h = None, None  # 합성 이미지의 최종 크기를 결정하기 위한 변수

        # 각 chunk 내의 모든 타겟 시간에 대해
        for dt in c_times:
            # 타겟 이미지 로드 (RGB)
            target_file = target_dict[dt]
            target_img = Image.open(target_file).convert("RGB")
            
            # 해당 시간 dt에 해당하는 예측 이미지 리스트 (없으면 건너뛰기)
            if dt not in pred_dict:
                continue
            pred_files_for_dt = pred_dict[dt]
            
            # 동일 시점에 대한 각 예측 이미지 순회
            for pred_file in pred_files_for_dt:
                if pred_file is None:
                    continue

                # (실제 코드 실행 딜레이가 필요하면 time.sleep() 사용 가능; MP4 생성에는 필요 없음)
                # time.sleep(1)
                
                pred_img = Image.open(pred_file).convert("RGB")
                w_pred, h_pred = pred_img.size
                w_tgt, h_tgt = target_img.size

                # 첫 프레임에서 합성 캔버스 크기 결정
                if out_w is None:
                    out_w = w_pred + w_tgt
                    out_h = max(h_pred, h_tgt) + 50  # 하단에 텍스트를 위한 여유 공간

                # 흰색 배경의 합성용 캔버스 생성
                combined = Image.new("RGB", (out_w, out_h), (255, 255, 255))
                combined.paste(pred_img, (0, 0))
                combined.paste(target_img, (w_pred, 0))

                # 텍스트 그리기
                draw = ImageDraw.Draw(combined)
                time_text = dt.strftime("%Y-%m-%d %H:%M")
                bbox = font.getbbox(time_text)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(((out_w - tw) / 2, out_h - th - 5),
                          time_text, fill="black", font=font)
                bbox2 = font.getbbox(date_range_text)
                trw, trh = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
                draw.text(((out_w - trw) / 2, out_h - th - trh - 30),
                          date_range_text, fill="black", font=font)
                
                # 최종 합성 이미지 numpy 배열로 변환 후 MP4 writer에 추가
                frame_array = np.array(combined)
                writer.append_data(frame_array)
        
        writer.close()
        print(f"Chunk {idx} 동영상이 생성되었습니다: {chunk_output_mp4}")

# =============================================================================
# 사용 예시:
# =============================================================================
# 예시: pred_dirs 리스트와 target_dir, 그리고 출력 MP4 파일의 기본 경로 지정
# pred_dirs = ["rs_pred_0", "rs_pred_1", "rs_pred_2", ...]
# target_dir = "rs_target_0"
# output_mp4 = "result.mp4"
# font_path = "path/to/font.ttf"  (옵션)

# 예시로 아래와 같이 호출:
# animation_image_newcode(["rs_pred_0", "rs_pred_1", "rs_pred_2"], "rs_target_0", "result.mp4", font_path="arial.ttf")

