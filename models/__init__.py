from models.segmentor.semalign import SemAlign

from models.backbone.text_encoder import CLIPTextEncoder
from models.backbone.img_encoder import CLIPVisionTransformer, VPTCLIPVisionTransformer
from models.decode_heads.semalign_head import SemAlignHead

from models.losses.atm_loss import SegLossPlus
from models.losses.rcs_loss import SegLossPlusRCS

from configs._base_.datasets.dataloader.voc12 import ZeroPascalVOCDataset20
from configs._base_.datasets.dataloader.coco_stuff import ZeroCOCOStuffDataset
from configs._base_.datasets.dataloader.context59 import ZeroPascalVOCDataset59
