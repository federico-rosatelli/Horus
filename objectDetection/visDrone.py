#pip install ultralyticsplus==0.0.28 ultralytics==8.0.43
from ultralyticsplus import YOLO, render_result

# load model
model = YOLO('mshamrai/yolov8n-visdrone')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000

image = "path_to_image"

results = model.predict(image)

print(results[0].boxes)
render = render_result(model=model, image=image, result=results[0])
render.show()