from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt

dataDir = "annotations_trainval2017/"
dataType = "val2017"
annFile = "{}/annotations/instances_{}.json".format(dataDir, dataType)

coco = COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]

catIds = coco.getCatIds(catNms=['person', 'bus'])
imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs(imgIds)[0]
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
for ann in anns:
    print(ann['segmentation'])

# I = io.imread(img['coco_url'])
# plt.axis('off')
# plt.imshow(I)
# coco.showAnns(anns)
# plt.show()