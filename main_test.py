import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.dataloader import *
from dataset.augmentations import *
from models.model import *
from utils import *
from torchvision import transforms
import torch
from models.model_DI import Resnet34_DI
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
import torch.nn.functional as F
import os
torch.cuda.set_device(3)


testroot = "single_orthogonal_mix/test"

# Build model
model = torch.load("checkpoints/best_model/mobilenet_v2_DI/30_tensor(100., device='cuda:3').pth")
model.cuda()
model.eval()
torch.no_grad()
print(model)
print('model load finish')
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.Resize(config.img_height, config.img_weight),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)
                                ])

accuracy = utils.accuracy
criterion = nn.CrossEntropyLoss().cuda()

def predict(model, testroot):
    model.cuda()
    model.eval()
    predlist = []
    labellist = []
    dirlist = []
    times = 0
    with torch.no_grad():
        for dir in os.listdir(testroot +'/' + 'crossed'):
            dirlist.append(int(dir))
            for img in os.listdir(testroot+ '/' + 'crossed' + '/' + dir):
                img1path = testroot + '/' + 'crossed' + '/' + dir + '/' + img
                img2path = testroot + '/' + 'single' + '/' + dir + '/' + img.split('_')[0] + '_single (' + img.split('(')[1]
                targets = Variable(torch.from_numpy(np.array([int(dir)])).long()).cuda()
                # img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)
                img1 = cv2.imread(img1path, -1)
                img2 = cv2.imread(img2path, -1)
                img1 = cv2.resize(img1, (320, 320))
                img2 = cv2.resize(img2, (320, 320))
                img1 = Image.fromarray(img1, mode="RGB")
                img2 = Image.fromarray(img2, mode="RGB")
                input1 = transform(img1)
                input2 = transform(img2)
                input1 = Variable(input1).cuda()
                input2 = Variable(input2).cuda()
                # input = input.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float32)
                input1 = input1.view(1,3,320,320)
                input2 = input2.view(1, 3, 320, 320)
                start = time.time()
                outputs = model(input1, input2)#
                end = time.time()
                _, pred = torch.max(outputs, dim=1)
                predlist.append(pred.cpu().numpy()[0])
                labellist.append(int(dir))
                t = end - start
                times += t
    fps = 1/(times/len(predlist))
    return predlist, labellist, dirlist, fps

predlist, labellist, dirlist, fps = predict(model, testroot)

print('fps:', fps)

namelist = ['Gra', 'Amb', 'Dia', 'Gab', 'Pe', 'V', 'T', 'Sil', 'L', 'Ph', 'Grg', 'Gag', 'Sem']
newlist = []
for num in dirlist:
    newlist.append(namelist[num])
print(predlist)
print('-----------')
print(labellist)
print('-----------')
print(newlist)
# Draw confusion matrix
def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None):
    # Using the functions in sklearn to generate confusion matrix and normalize it
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    zhfont1 = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/dejavu/TIMESBD.TTF')
    zhfont2 = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/dejavu/SIMSUN.TTC')

    # Draw a picture. If you want to change the color style, you can change this part's cmap=pl.get_ At cmap ('Blues')
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Reds'))
    plt.colorbar()

    # Image Title
    if title is not None:
        plt.title(title, fontproperties=zhfont2,fontsize=10)
    # Draw Coordinates
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name

    plt.xticks(num_local, axis_labels, rotation=90, fontproperties=zhfont1, fontsize=10)
    plt.yticks(num_local, axis_labels, fontproperties=zhfont1, fontsize=10)
    plt.ylabel('True label',fontproperties=zhfont2)
    plt.xlabel('Prediction label',fontproperties=zhfont2)

    # Print the percentage in the corresponding box. White words are used for those greater than threshold, and black words are used for those less than threshold
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                         fontproperties=zhfont1,
                         fontsize=8,
                         ha="center", va="center",
                         color="white" if cm[i][j] > thresh else "black")
plot_matrix(np.array(labellist), np.array(predlist), labels_name = dirlist,title = 'Confusion Matrix', axis_labels = newlist)
plt.tight_layout()
plt.savefig("map.png", dpi=300)