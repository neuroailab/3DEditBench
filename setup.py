from setuptools import setup, find_packages

setup(
    name="edit_bench",
    version="0.1",
    packages=find_packages(),
    description="3DEditBench: Object editing and evaluation benchmark",
    author="Stanford NeuroAI",
    install_requires=[
        "lpips",
        "segment_anything",
        "ptlflow",
        "ipykernel",
        "h5py",
        "numpy==1.26.4",
        "torch==2.1.2",
        "torchvision",
        "scikit-image",
        "matplotlib",
        "opencv-python",
        "pillow",
        "tqdm",
        "requests==2.32.4",
        "google-cloud-storage",
        "imageio",
        "moviepy",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "editbench-infer = edit_bench.run_inference:main",
            "editbench-launch = edit_bench.run_inference_parallel:main",
            "editbench-evaluate-metrics=edit_bench.evaluate_folder:main"
        ]
    }
)
