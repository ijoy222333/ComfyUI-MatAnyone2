\# ComfyUI-MatAnyone2



A custom node wrapper for the official MatAnyone2 repository.



\## Important



This project is only a ComfyUI wrapper.



You must also clone the official MatAnyone2 repository into:



```text


在你的ComfyUI\custom_nodes\ComfyUI-MatAnyone2目录新建third_party文件夹，然后运行
git clone https://github.com/pq-yang/MatAnyone2.git
下载MatAnyone2，然后进入MatAnyone2文件夹
X:\你的python环境\python.exe -m pip install -e .
cd /d X:\ComfyUI\custom_nodes\ComfyUI-MatAnyone2\third_party\MatAnyone2
F:\你的python环境\python.exe -m pip install -e .
如果安装报错可以进入：F:\ComfyUI\custom_nodes\ComfyUI-MatAnyone2\third_party\MatAnyone2\pyproject.toml
找到依赖列表里的这行：
'cchardet >= 2.1.7',
'PySide6 >= 6.2.0',
'pyqtdarktheme',
这3个都删除在安装依赖即可。
模型MEDEL目录：X:\ComfyUI\models\MatAnyone2\matanyone2.pth
download：
https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth
最后pip install -r requirements.txt



