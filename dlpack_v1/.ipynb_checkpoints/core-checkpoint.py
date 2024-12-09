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

    # GIF 생성
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
                writer.append_data(frame_array, duration=frame_duration)


            # 한 덩어리 끝나면 10초 공백 프레임 삽입
            pause_frame = Image.new("RGB", (out_w, out_h), (255, 255, 255))
            writer.append_data(imageio.v3.imread(pause_frame), duration=10.0)

    print(f"Animation saved as {output_gif}")




