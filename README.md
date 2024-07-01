# Tracking 

## 설치
1. 가상환경 생성
    ```sh
    conda create -n tracker python=3.8
    conda activate tracker
    ```
2. ByteTrack 설치
    ```sh
    git clone https://github.com/ifzhang/ByteTrack.git
    cd ByteTrack
    pip3 install -r requirements.txt 
    python3 setup.py develop

    pip3 install cython; 
    pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    
    pip3 install cython_bbox
    ```
    설치 과정 중 ERROR: No matching distribution found for onnxruntime==1.8.0 발생하면 [참고](https://stackoverflow.com/questions/72127093/i-am-unable-to-install-onnxruntime-with-pip3-please-resolve-it)

3. Ultralytics 설치
    ```
    pip install ultralytics==8.0.10
    ```

4. Roboflow Supervision 설치
    ```
    pip install supervision==0.1.0
    ```
    ```
    pip install onemetric
    ```

5. yolo 실행 시 error
    ```
    pip install protobuf==3.20.*
    protobuf-5.27.1

    pip install numpy==1.23.4
    ```