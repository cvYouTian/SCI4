from pathlib import Path
from utils.data import *
from utils.metric import *
from argparse import ArgumentParser
import torch.utils.data as Data
from model.TGRSNet import *
from model.loss import *
from torch.optim import Adagrad
from tqdm import tqdm
import os.path as osp
import os
import time

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    print("Warning: thop not installed,Please install it with pip install thop")
    THOP_AVAILABLE = False


def seed_pytorch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保持性能的设置
    torch.backends.cudnn.benchmark = True  # 自动优化
    # 不设置 deterministic=True 和 num_threads=1


def parse_args():
    parser = ArgumentParser(description='Implement of model')
    # 设置数据集的绝对路径
    parser.add_argument('--dataset-dir', type=str,
                        default='/home/tian/Documents/datasets/IRSTD-1k')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--warm-epoch', type=int, default=5)
    parser.add_argument('--start_epoch', type=int, default=0)

    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--crop-size', type=int, default=256)

    parser.add_argument('--multi-gpus', type=bool, default=False)
    parser.add_argument('--if-checkpoint', type=bool, default=False)

    parser.add_argument('--mode', type=str, default='train')
    # parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--weight-path', type=str, default='./weight/prom4-table19-hang3/best.pkl')

    args = parser.parse_args()
    return args


class Get_gradient_nopadding(nn.Module):
    """提取边缘特征"""

    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)

        if torch.cuda.is_available():
            self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
            self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()
        else:
            self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cpu()
            self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cpu()

    def forward(self, x):
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]

        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)
        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)
        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        return torch.cat([x0, x1, x2], dim=1)


class Get_gradientmask_nopadding(nn.Module):
    """提取mask的边缘特征"""

    def __init__(self):
        super(Get_gradientmask_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)

        if torch.cuda.is_available():
            self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
            self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()
        else:
            self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cpu()
            self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cpu()

    def forward(self, x):
        
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)
        e = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        binary = (e > 0.5).float()
        return binary


class Trainer:
    def __init__(self, args):
        assert args.mode == 'train' or args.mode == 'test', "setting your mode please"
        self.args = args
        self.start_epoch = args.start_epoch
        self.mode = args.mode

        trainset = IRSTD_Dataset(args, mode='train')
        valset = IRSTD_Dataset(args, mode='val')

        self.train_loader = Data.DataLoader(trainset, args.batch_size, shuffle=True, drop_last=True)
        self.val_loader = Data.DataLoader(valset, 1, drop_last=False)

        device = torch.device('cuda')
        self.device = device

        self.grad = Get_gradient_nopadding()
        self.grad_mask = Get_gradientmask_nopadding()

        model = TGRS(3)

        if args.multi_gpus:
            if torch.cuda.device_count() > 1:
                print('use ' + str(torch.cuda.device_count()) + ' gpus')
                model = nn.DataParallel(model, device_ids=[0, 1])
        model.to(device)
        self.model = model

        self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)

        self.down = nn.MaxPool2d(2, 2)
        self.loss_fun = SLSIoULoss()
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        self.ROC = ROCMetric(1, 10)
        self.best_iou = 0
        self.warm_epoch = args.warm_epoch

        if args.mode == 'train':
            if args.if_checkpoint:
                check_folder = ''
                checkpoint = torch.load(check_folder + '/checkpoint.pkl')
                self.model.load_state_dict(checkpoint['net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_iou = checkpoint['iou']
                self.save_folder = check_folder
            else:
                self.save_folder = '/home/tian/Documents/development/repos/TGRS2025/weight/Net-%s' % (
                    time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
                self.save_folder = Path(self.save_folder)
                if not self.save_folder.exists():
                    self.save_folder.mkdir(exist_ok=True, parents=True)

        if args.mode == 'test':
            weight = torch.load(args.weight_path, map_location="cpu")
            self.model.load_state_dict(weight)
            self.warm_epoch = -1

        self.print_iniitialization_info()

    def print_iniitialization_info(self):
        """
        打印初始化信息
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f'*' * 80)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        if THOP_AVAILABLE:
            try:
                # 创建输入
                input_shape = (1, 3, self.args.crop_size, self.args.crop_size)

                # 创建dummy输入
                dummy_input = torch.randn(input_shape)

                # 将模型设置成评估模式并移动到CPU上进行计算
                model_copy = type(self.model.module if hasattr(self.model, "module") else self.model)(3)
                if hasattr(self.model, "module"):
                    model_copy.load_state_dict(self.model.module.state_dict())
                else:
                    model_copy.load_state_dict(self.model.state_dict())
                model_copy.eval()

                # 计算FLOPs和参数量
                flops, params = profile(model_copy, inputs=(dummy_input, False))
                flops, params = clever_format([flops, params], "%.3f")

                print(f"GFLOPs: {flops}")
                print(f"Params (from thop): {params}")
            except Exception as e:
                print(f"Failed to calculate GFLOPs: {str(e)}")
                print("This might be due to model architecture complexity.")
        else:
            print("GFLOPs: N/A (thop not installed)")

        print(f'*' * 80 + "\n")

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()
        tag = False
        for i, (data, mask, edge_ori) in enumerate(tbar):
            data = data.to(self.device)
            labels = mask.to(self.device)

            if epoch > self.warm_epoch:
                tag = True

            masks, pred = self.model(data, tag)
            loss = 0

            loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch)
            for j in range(len(masks)):
                if j > 0:
                    labels = self.down(labels)
                loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch)

            loss = loss / (len(masks) + 1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))

    def test(self, epoch):
        self.model.eval()

        # 指标初始化
        self.mIoU.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()
        self.ROC.reset()

        tbar = tqdm(self.val_loader)
        tbar.set_description('Epoch %d' % (epoch))

        tag = False
        update_freq = max(1, min(len(self.val_loader) // 10, 10))

        for i, (data, mask, _) in enumerate(tbar):
            with torch.no_grad():
                data = data.to(self.device)
                mask = mask.to(self.device)

                if epoch > self.warm_epoch:
                    tag = True

                _, pred = self.model(data, tag)

            self.mIoU.update((pred > 0.5).float(), mask)
            self.nIoU_metric.update((pred > 0.5).float(), mask)
            self.PD_FA.update((pred > 0.5).float(), mask)
            self.ROC.update((pred > 0.5).float(), mask)

            if i % update_freq == 0:
                FA, PD = self.PD_FA.get(len(self.val_loader))
                _, IoU = self.mIoU.get()
                _, nIoU = self.nIoU_metric.get()

                tbar.set_postfix({
                    'IoU': f"{IoU:.4f}",
                    'nIoU': f"{nIoU:.4f}",
                    'PD': f"{float(PD[0]):.4f}",
                    'FA': f"{FA[0] * 1e6:.2f}"
                })

        FA, PD = self.PD_FA.get(len(self.val_loader))
        _, IoU = self.mIoU.get()
        _, nIoU = self.nIoU_metric.get()
        ture_positive_rate, falsepositive_rate, _, _ = self.ROC.get()

        if self.mode == 'train':
            self.save_folder = str(self.save_folder)
            if IoU > self.best_iou:
                self.best_iou = IoU
                torch.save(self.model.state_dict(), self.save_folder + '/best.pkl')
                # 保存最好指标
                with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                    f.write(
                        '{} - {:04d}\t - best_IoU {:.4f}\t - current_nIoU {:.4f}\t - current_PD {:.4f}\t - current_FA {:.4f}\n'.
                        format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                               epoch, IoU, nIoU, PD[0], FA[0] * 1e6))

            # 保存训练日志
            with open(osp.join(self.save_folder, 'train.log'), 'a') as f:
                f.write(
                    '{} - {:04d}\t - current_IoU {:.4f}\t - current_nIoU {:.4f}\t - current_PD {:.4f}\t - current_FA {:.4f}\n'.
                    format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                           epoch, IoU, nIoU, PD[0], FA[0] * 1e6))

            torch.save(self.model.state_dict(), self.save_folder +'/last.pkl')

        elif self.mode == 'test':
            print('IoU: ' + str(IoU) + '\n')
            print("nIoU: ," + str(nIoU) + '\n')
            print('Pd: ' + str(PD[0]) + '\n')
            print('Fa: ' + str(FA[0] * 1e6) + '\n')


if __name__ == '__main__':
    args = parse_args()
    seed_pytorch()

    trainer = Trainer(args)

    if trainer.mode == 'train':
        for epoch in range(args.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)
    else:
        trainer.test(1)
