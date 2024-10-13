from setuptools import setup

setup(
    name='DataFlow',
    version='0.1.0',
    py_modules=[],
    install_requires=['av==12.3.0', 'datasets==2.21.0', 'decord==0.6.0', 'jsonargparse==4.32.0', 'numpy==1.26.4', 'opencv-python==4.10.0.84', 'scipy==1.13.1', 'torch==2.4.0', 'tqdm==4.66.5', 'transformers==4.44.2'],
    extras_require={
        'image': ['fsspec==2024.6.1', 'ftfy==6.2.3', 'nltk==3.8', 'openai-clip==1.0.1', 'regex==2024.7.24', 'safetensors==0.4.4', 'scikit-learn==1.5.1', 'setuptools==72.1.0', 'timm==1.0.8', 'torchvision==0.19.0', 'vllm==0.6.0'],
        'video': ['einops==0.8.0', 'pandas==2.2.2', 'PyYAML==6.0.2', 'scikit-video==1.1.11', 'timm==1.0.8', 'torchvision==0.19.0'],
        'text': ['fasttext==0.9.3', 'filelock==3.15.4', 'google-api-core==2.19.1', 'google-api-python-client==2.140.0', 'google-auth==2.33.0', 'google-auth-httplib2==0.2.0', 'googleapis-common-protos==1.63.2', 'kenlm==0.2.0', 'langkit==0.0.33', 'loguru==0.7.2', 'matplotlib==3.9.2', 'multiprocess==0.70.16', 'openai==1.44.1', 'prettytable==3.11.0', 'pyspark==3.5.2', 'sentencepiece==0.2.0', 'vendi-score==0.0.3', 'wget==3.2', 'nltk','presidio_analyzer[transformers]','presidio_anonymizer'],
        'all': ['fsspec==2024.6.1', 'ftfy==6.2.3', 'nltk==3.8', 'openai-clip==1.0.1', 
            'regex==2024.7.24', 'safetensors==0.4.4', 'scikit-learn==1.5.1', 
            'setuptools==72.1.0', 'timm==1.0.8', 'torchvision==0.19.0', 'vllm==0.6.0',
            'einops==0.8.0', 'pandas==2.2.2', 'PyYAML==6.0.2', 'scikit-video==1.1.11',
            'timm==1.0.8', 'torchvision==0.19.0', 'fasttext==0.9.3', 'filelock==3.15.4',
            'google-api-core==2.19.1', 'google-api-python-client==2.140.0', 
            'google-auth==2.33.0', 'google-auth-httplib2==0.2.0', 
            'googleapis-common-protos==1.63.2', 'kenlm==0.2.0', 'langkit==0.0.33',
            'loguru==0.7.2', 'matplotlib==3.9.2', 'multiprocess==0.70.16', 
            'openai==1.44.1', 'prettytable==3.11.0', 'pyspark==3.5.2', 
            'sentencepiece==0.2.0', 'vendi-score==0.0.3', 'wget==3.2'],
    }
)