from setuptools import setup, find_packages

setup(
    name='deep_marker_estimation',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        # 'opencv-python',
        'scikit-image',
        'matplotlib',
        'Pillow',
        'tqdm',
        'albumentations',
        'scipy',
        'pyyaml',
        'rospkg',  # For ROS Python integration (optional)
    ],
    entry_points={
        'console_scripts': [
            # You can define command-line entry points here if desired
            # 'estimate_marker = deep_marker_estimation.inference:main',
        ],
    },
    author='Your Name',
    author_email='your@email.com',
    description='Deep learning-based marker pose estimation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/deep-marker-estimation',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    license='MIT',
    python_requires='>=3.6',
    zip_safe=False,
)
