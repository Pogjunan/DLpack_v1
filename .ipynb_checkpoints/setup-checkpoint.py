# my_utils/setup.py
from setuptools import setup, find_packages

setup(
    name="dlpack_v1",
    version="1.0",
    packages=find_packages(),
    description="A package for preprocessing or offering preprocessing tools",
    author="junhan Jeon",
    author_email="jjh15964@naver.com", #ref : guebin@jbnu.ac.kr
    install_requires=[],  # 필요한 패키지 목록이 있다면 추가
)
