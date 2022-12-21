import warnings
from torch import nn,optim
from torchvision import transforms
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from dataset.mydataset import *
from dataset.dataloader import *
# from models.model import *
from models.model_DI import *
from utils import *
# from visualizations.vis import Visualizer
from dataset.config import config
torch.cuda.set_device(3)


#1. set random.seed and cudnn performanceXX
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
# def net(num_classes):


#2. evaluate func
def evaluate(val_loader,model,criterion,epoch):
    #2.1 define meters
    losses = AverageMeter()
    top1 = AverageMeter()
    #progress bar
    val_progressor = ProgressBar(mode="Val  ",
                                 epoch=epoch,
                                 total_epoch=config.epochs,
                                 model_name=config.model_name,total=len(val_loader))
    #2.2 switch to evaluate mode and confirm model has been transfered to cuda
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i,batches in enumerate(val_loader):
            val_progressor.current = i
            # Define input image
            input1 = Variable(batches['image1']).cuda()
            input2 = Variable(batches['image2']).cuda()
            # Define dimension information
            target = Variable(torch.from_numpy(np.array(batches['label'])).long()).cuda()
            # Neural network output
            output = model(input1, input2)
            loss = criterion(output,target)

            #2.2.2 measure accuracy and record loss
            precision1,precision2 = accuracy(output,target,topk=(1,2))
            losses.update(loss.item(),input1.size(0))
            top1.update(precision1[0],input1.size(0))


            val_progressor.current_loss = losses.avg
            val_progressor.current_top1 = top1.avg
            val_progressor()
        val_progressor.done()
    return [losses.avg,top1.avg]


def main():
    fold = 0
    #4.1 mkdirs
    if not os.path.exists(config.submit):
        os.mkdir(config.submit)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold) + os.sep):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold) + os.sep)
    if not os.path.exists(config.best_models + config.model_name + os.sep +str(fold) + os.sep):
        os.makedirs(config.best_models + config.model_name + os.sep +str(fold) + os.sep)


    # vis = Visualizer(env=config.model_name)
    # Build model
    model = eval(config.model_name)(n_classes=config.num_classes, pretrained=True)
    # model = densenet121(n_classes=config.num_classes, pretrained=True)
    # model = inceptionV3(n_classes=config.num_classes, pretrained=True)
    # model = Resnet101(n_classes=config.num_classes, pretrained=True)
    model.cuda()


    optimizer = optim.Adam(model.parameters(),
                           lr = config.lr,
                           amsgrad=True,
                           weight_decay=config.weight_decay)
    # Define cross entropy loss function
    criterion = nn.CrossEntropyLoss().cuda()

    start_epoch = 0
    best_precision1 = 0
    best_precision_save = 0

    # Read Data
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.Resize(config.img_height, config.img_weight),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)
                                    ])
    train_dataloader = DataLoader(myDataset_DI(imgroot = config.dataroot, n_class=config.num_classes, width= config.img_weight, height= config.img_height, data = 'train', transform = transform,target_transform=False),
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  # collate_fn=collate_fn,
                                  pin_memory=True,
                                  num_workers=0)
    val_dataloader = DataLoader(myDataset_DI(imgroot = config.dataroot, n_class=config.num_classes, width= config.img_weight, height= config.img_height, data = 'test', transform = transform,target_transform=False),
                                batch_size=config.batch_size*2,
                                shuffle=True,
                                # collate_fn=collate_fn,
                                pin_memory=False,
                                num_workers=0)

    scheduler =  optim.lr_scheduler.StepLR(optimizer,
                                           step_size = 10,
                                           gamma=0.1)
    #4.5.5.1 define metrics
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    valid_loss = [np.inf,0,0]
    model.train()

    #4.5.5 train
    for epoch in range(start_epoch,config.epochs):
        scheduler.step(epoch)
        # Define Progress Bar
        train_progressor = ProgressBar(mode="Train",epoch=epoch,
                                       total_epoch=config.epochs,
                                       model_name=config.model_name,
                                       total=len(train_dataloader))

        # train
        # for iter,(input1, input2, target) in enumerate(train_dataloader):
        for iter, batches in enumerate(train_dataloader):
            # print(batches)
            train_progressor.current = iter
            model.train()

            # Define input image
            input1 = Variable(batches['image1']).cuda()
            input2 = Variable(batches['image2']).cuda()
            # Define dimension information
            target = Variable(torch.from_numpy(np.array(batches['label'])).long()).cuda()
            # Neural network output
            output = model(input1, input2)
            # output = model.fc(output)#densenet
            # Calculating losses
            loss = criterion(output,target)

            precision1_train,precision2_train = accuracy(output,target,
                                                         topk=(1,2))
            train_losses.update(loss.item(),input1.size(0))
            train_top1.update(precision1_train[0],input1.size(0))

            train_progressor.current_loss = train_losses.avg
            train_progressor.current_top1 = train_top1.avg

            #if (iter + 1) % config.plot_every == 0:

            # vis.plot('train_loss', train_losses.avg)
            # vis.plot('train_precision', train_top1.avg)

            # Gradient back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Show Progress Bar
            train_progressor()

        train_progressor.done()


        #evaluate
        #lr = get_learning_rate(optimizer)

        #evaluate every half epoch
        valid_loss = evaluate(val_dataloader,model,criterion,epoch)
        # torch.save(model, config.best_models + config.model_name + '/' + str(epoch+1) + '_' + str(valid_loss[1]) + '.pth')
        is_best = valid_loss[1] > best_precision1
        if is_best:
            torch.save(model, config.best_models + config.model_name + '/' + str(epoch + 1) + '_' + str(
                valid_loss[1]) + '.pth')


if __name__ =="__main__":
    main()





















