"""pip package: qwenfpt-tts

CLI entry point:
    tts --voice ryan --text "Hello world" --output out.wav

Inference package deliberately excludes nemo_text_processing.
Runtime text normalization uses num2words + regex only.

Package name: qwenfpt-tts (fallback from custom-en-tts if PyPI name taken).
Check PyPI for conflicts before publishing. Publish to TestPyPI first.
"""

from setuptools import setup, find_packages

setup(
    name="qwenfpt-tts",
    version="0.1.0",
    description="Custom English TTS — 6 voices trained on LibriSpeech 100h (Qwen3-TTS architecture)",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "torchaudio>=2.3.0",
        "transformers>=4.42.0",
        "numpy>=1.26.0",
        "soundfile>=0.12.0",
        "num2words>=0.5.13",  # runtime text normalization — lightweight
    ],
    entry_points={
        "console_scripts": [
            "tts=inference.cli:main",  # implement in Phase 7
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",  # update per D5 decision
        "Operating System :: OS Independent",
    ],
)
