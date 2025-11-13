
# 使用screen运行后台进程（守护进程）

因为ssh等属于前台进程，故而需要screen将当前bash切换为daemon（守护进程）

    apt install screen

如上，安装了screen后，如果要运行类似于`yolo train ... epoch=150`之类的很长时间的命令，人会离开电脑、网络不稳定导致的断网等因素，前台进程通常不会成功运行好几个小时的进程

所以使用`screen`来运行此类命令

    screen -S <screen name>

在screen新打开的终端输入`yolo train ... epoch=150`这种长时间的前台命令

然后**ctrl+A+D**推出main1的**daemon terminal**

`screen -ls`查看你的screen daemon

```bash
(main) root@C.27696447:/workspace$ screen -ls
There are screens on:
        3883.main1      (11/13/25 10:25:13)     (Detached)
        1769.main       (11/13/25 10:07:52)     (Attached)
2 Sockets in /run/screen/S-root.
```

`screen -r <pid>或者<screen name>`回到之前创建的screen 

## 复现工作全部记录
安装：pip install -U "ultralytics>=8.3.200" opencv-python tqdm numpy matplotlib torch torchvision

```bash
cd /workspace/
git clone https://github.com/SenRanja/Aug.git
pip install albumentations kagglehub
python ./Aug/download.py
```

可以看到数据集在`/root/.cache/kagglehub/datasets/rupankarmajumdar/crop-pests-dataset/versions/2/`

下一步我们把这些东西迁移到workspace目录里面，这样我认为更符合操作习惯
```bash
cd /workspace

mkdir -p crop_pests

ln -s /root/.cache/kagglehub/datasets/rupankarmajumdar/crop-pests-dataset/versions/2 \
      crop_pests/data

```

总之，这一步要确保你的workspace里面有两个大文件夹。分别是`Aug`,和`crop_pest/data`。`crop_pest/data`里面有 `train`, `valid`, `test`, `rtdetr_repro.py`, `predict_safe.py`, `data.yaml`这些文件

### 我们首先运行数据增强任务
```bash
cd /workspace/Aug
python main.py /workspace/crop_pest/data
```
*需要注意的是，这里的参数就是train文件夹的子目录，用户自我检查一下，输入一定是绝对路径*


#### 运行脚本`repro_detr_l.py`
**注意这里，你要注意有没有在data.yaml上用绝对路径，相对路径可能报错**

```bash
cd /root/.cache/kagglehub/datasets/rupankarmajumdar/crop-pests-dataset/versions/2/
python rtdetr_repro.py train \
  --model rtdetr-l.pt \
  --data data.yaml \
  --device 0 \
  --epochs 150 \
  --imgsz 600 \
  --batch 24 \
  --workers 4 \
  --cos_lr \
  --cache \
  --save_dir /workspace/rtdetr_aug_b8_600 > /workspace/train.log 2>&1
```
下面是最近一次的运行指令，截止运行4小时后，没有发生早停现象。理由是mAP90-95一直在变好，只要这一项在优化，就不会发生停止
```bash
python rtdetr_repro.py train \
  --model rtdetr-l.pt \
  --data data.yaml \
  --device 0 \
  --epochs 150 \
  --imgsz 600 \
  --batch 24 \
  --workers 4 \
  --cos_lr \
  --cache \
  --patience 30 \
  --save_dir rtdetr_data_enhance_600img_150e

```


