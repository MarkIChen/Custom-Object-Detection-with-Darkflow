import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import os

options = {
    'model': 'cfg/yolov2-tiny-voc-1c.cfg',
    'load': 5000,
    'threshold': 0.1,
}

tfnet = TFNet(options)


# read the color image and covert to RGB
imgs =[]
folder = 'test/Winnie'
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

print("There are " + str(len(imgs))+ " images in the folder.")

# pull out some info from the results

for img in imgs:
    results = tfnet.return_predict(img)
    print(results)

    for result in results:
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        label = result['label']
        confidence = result['confidence']
        text = '{}: {:.0f}%'.format(label, confidence * 100)

        img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
        img = cv2.putText(img, text, tl, cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 15)
    
    plt.imshow(img)
    plt.show()