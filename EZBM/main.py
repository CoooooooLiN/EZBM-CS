# coding:utf-8
import os, time, random, torch, argparse, warnings
import utils
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataset import IMBALANCECIFAR10, IMBALANCECIFAR100, CINIC10
from model import EZBM


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', default='cinic10', help='dataset setting: cifar10/cifar100/cinic10')
parser.add_argument('--model_name', default='resnet32', type=str, help='model name')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.02, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default=None, type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')

parser.add_argument('--num_workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--use_norm', default=False, type=bool, help='use norm')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--expansion_mode', default='orginal', type=str, help='orginal/balance/reverse')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--resume', default=False, type=str,  help='path to latest checkpoint (default: none)')

parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
parser.add_argument('--device', default='cuda', type=str, help='use GPU.')


def main():
    args = parser.parse_args()
    # args.seed = random.randint(1, 10000)
    # prepare related documents
    if not os.path.exists('log'):
        os.makedirs('log')
    log_dir = args.dataset + '_' + args.imb_type + '_' + str(args.imb_factor) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join('log', log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    utils.configure_output_dir(log_dir)
    hyperparams = dict(args._get_kwargs())
    utils.save_hyperparams(hyperparams)

    # if args.seed is not None:
    #     random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     np.random.seed(args.seed)

    # prepare related data
    print("=> preparing data sets: {}, imablanced ratio: {}, type: {}"
          .format(args.dataset, args.imb_factor, args.imb_type))
    num_classes = 100 if args.dataset == 'cifar100' else 10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root='../DataSet', imb_type=args.imb_type, imb_factor=args.imb_factor,
                                         rand_number=args.rand_number, train=True, download=True,
                                         transform=transform_train)
        val_dataset = datasets.CIFAR10(root='../DataSet', train=False, download=True, transform=transform_val)
    if args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root='../DataSet', imb_type=args.imb_type, imb_factor=args.imb_factor,
                                          rand_number=args.rand_number, train=True, download=True,
                                          transform=transform_train)
        val_dataset = datasets.CIFAR100(root='../DataSet', train=False, download=True, transform=transform_val)
    if args.dataset == 'cinic10':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]),
        ])
        train_dataset = CINIC10(root='../DataSet/cinic10', imb_type=args.imb_type,
                           imb_factor=args.imb_factor, rand_number=10, type='train', transform=transform_train)
        val_dataset = CINIC10(root='../DataSet/cinic10', type='valid', transform=transform_val)

    cls_num_list = train_dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)

    # initialize model
    use_norm = True if args.use_norm else False
    model = EZBM(args, cls_num_list, num_classes)

    # start training
    # utils.load_pytorch_model(model)
    if not args.resume:
        model.fit(train_dataset, args.epochs)
        utils.save_pytorch_model(model)
    else:
        utils.load_pytorch_model(model)

    # start testing
    test_accuracy = model.predict(val_dataset)

    # record classification results
    result_file = open(os.path.join(log_dir, "result.txt"), 'w')
    result_file.write(np.array2string(test_accuracy))
    result_file.write("\n")
    result_file.write(np.array2string(np.mean(test_accuracy)))
    result_file.close()


if __name__ == '__main__':
    main()