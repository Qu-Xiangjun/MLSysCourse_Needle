import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="needleTensor",
    version="0.0.1",
    author="Xiangjun Qu",
    author_email="quxiangjun@mail.ustc.edu.cn",
    description="A mini pytorch, which contains necessary element if deep learning framwork.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Xiangjun-Qu/NeedleTensor",
    packages=setuptools.find_packages("needle"),
    install_requires=['numpy'],
    package_data={
        'needle.backend_ndarray': [],  # 包含所有 .so 文件
    },
    classifiers = [
    "Development Status :: 3 - Alpha",           # 开发状态 (Alpha, Beta, Production/Stable)
    "Intended Audience :: Developers",           # 预期的用户群
    "License :: OSI Approved :: MIT License",    # 许可证
    "Programming Language :: Python :: 3",       # 支持的 Python 版本
    "Programming Language :: Python :: 3.8",     # 具体支持的 Python 版本
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Software Development :: Libraries", # 主题/领域
    "Topic :: Utilities",
    ]
)