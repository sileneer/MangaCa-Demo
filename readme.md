##Requirements:

1. Python 3.7
2. CUDA Runtime 10.0 (https://developer.nvidia.com/cuda-10.0-download-archive)
3. Windows OS (Linux also can but need change line 341 in `mangaca_server.py` to `tmp_path = 'tmp/mangaca_%d.%s' % (random.randint(0, 10000000), ext)`)
4. `!pip install -r requirements.txt`
5. Run `mangaca_server.py`

Take note of this line of console log when you run `mangaca_server.py`: `Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)`, fill in the address here into line 138 of `index.html`: `var apiURLs = ['whatever link you see/api']`, remember to add '/api' to the end of the link.
E.g. `var apiURLs = ['http://127.0.0.1:5000/api']`

All credits of this work goes to the original author: https://github.com/Zerui18/MangaCa-Demo