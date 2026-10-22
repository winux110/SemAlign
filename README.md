# SemAlign

Code for **Semantic Calibration and Decoder Alignment for One-Stage Zero-Shot Semantic Segmentation**, built on [Cascade-CLIP](https://github.com/HVision-NKU/Cascade-CLIP) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

## Installation

Targets Linux with NVIDIA GPUs and the OpenMMLab stack:

```bash
conda create -n semalign python=3.7 -y
conda activate semalign
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install openmim
mim install mmcv-full==1.5.0
pip install mmsegmentation==0.24.0
pip install -r requirements.txt
```

Place the OpenAI CLIP ViT-B/16 checkpoint at `clip_pretrain/ViT-B-16.pt`.

Prepare PASCAL VOC 2012 under `datasets/VOCdevkit/VOC2012` and COCO-Stuff 164K under `datasets/coco_stuff164k`.

## Usage

```bash
bash scripts/dist_train.sh configs/semalign_voc_snapshot.py 2 work_dirs/voc --seed 0
bash scripts/dist_train.sh configs/semalign_coco_snapshot.py 2 work_dirs/coco --seed 0
python test.py configs/semalign_voc_snapshot.py work_dirs/voc/latest.pth --eval mIoU
bash scripts/dist_test.sh configs/semalign_coco_snapshot.py work_dirs/coco/latest.pth 2 --eval mIoU
```

Use `CUDA_VISIBLE_DEVICES` to select GPUs.

## License

MIT, see [LICENSE](LICENSE).
