import argparse
import os
import torch
import ipdb
from exp.exp_MSTFormer import Exp_MSTFormer
import log as L

# argparse 模块是 Python 内置的一个用于命令项选项与参数解析的模块，argparse 模块可以让人轻松编写用户友好的命令行接口。
# 通过在程序中定义好我们需要的参数，然后 argparse 将会从 sys.argv 解析出这些参数。argparse 模块还会自动生成帮助和使用手册，
# 并在用户给程序传入无效参数时报出错误信息。

# ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
parser = argparse.ArgumentParser(description='[MSTFormer] Long Sequences Forecasting')

# 给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的。通常，
# 这些调用指定 ArgumentParser 如何获取命令行字符串并将其转换为对象。这些信息在 parse_args() 调用时被存储和使用。
# choices - 参数可允许的值的一个容器。 required - 可选参数是否可以省略 (仅针对可选参数)。
parser.add_argument('--model', type=str, default='MSTFormer',
                    help='model of experiment, options: [MSTFormer]')
# default - 不指定参数时的默认值。 type - 命令行参数应该被转换成的类型
parser.add_argument('--data', type=str, default='ship', help='data')
parser.add_argument('--root_path', type=str, default='./data/AIS2021_area_17_31_-98.5_-80/',
                    help='root path of the data file')
parser.add_argument('--data_type', type=str, default=['ALL'], help='choose ship type, options: [ALL, [type_num]]')
parser.add_argument('--data_path', type=str, default='ALL', help='0.0/357015000.npy')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='t',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=72, help='input sequence length of MSTFormer encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token length of MSTFormer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
parser.add_argument('--prob_use', type=str, default=True, help='whether use probility map')
# MSTFormer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

# 更改data_parser中的参数M
parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=2, help='decoder input size')
parser.add_argument('--c_out', type=int, default=2, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=3, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
# action - 命令行遇到参数时的动作，默认值是 store。
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
# action='store_true'：如果在跑代码时，触发'--output_attention'就为true
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=50, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=5e-6, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
# 想要获得最终预测的话这里应该设置为True；否则将是获得一个标准化的预测 action='store_true'
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')#'0,1,2,3'
parser.add_argument('--read_processed', type=str, default='./data/AIS2021_72_48_24_3s_test/',
                    help='path to the processed data')
# ./data/AIS2021_72_48_24_3s_delta/
# ArgumentParser 通过 parse_args() 方法解析参数。它将检查命令行，把每个参数转换为适当的类型然后调用相应的操作。
# 在大多数情况下，这意味着一个简单的 Namespace 对象将从命令行解析出的属性构建
args = parser.parse_args()

# 是否使用GPU
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
# args.use_gpu = False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ship': {'data': 'ALL', 'T': 'Heading', 'M': [2, 2, 2], 'S': [1, 1, 1], 'MS': [3, 3, 1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']  # target feature in S or MS task，预测问题为M时表示多个sequence预测多个sequence
    args.enc_in, args.dec_in, args.c_out = data_info[
        args.features]  # encoder input size，decoder input size，output size

args.detail_freq = args.freq  # freq for time features encoding 当flag=='pred'
args.freq = args.freq[-1:]
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


print('Args in experiment:')
print(args)

Exp = Exp_MSTFormer
s = 'model:{}_data:{}_probUse:{}_embed:{}_encoderLayers:{}_decoderLayers:{}_probSparseFactor:{}_dropout:{}_itr:{}_trainEpochs:{}_batchSize:{}_patience:{}_learningRate:{}'.format(
    args.model, args.read_processed, args.prob_use, args.embed, args.e_layers, args.d_layers, args.factor,
    args.dropout,
    args.itr, args.train_epochs, args.batch_size, args.patience, args.learning_rate)
L.write_log(s)
data_name = 'model1'
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_pu{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
        data_name, args.model, args.data, args.features,
        args.seq_len, args.label_len, args.pred_len, args.prob_use,
        args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor,
        args.embed, args.distil, args.mix, args.des, ii)
    exp = Exp(args)  # set experiments
    s = "itr:{},total:{}\n".format(ii, args.itr)
    s = s + '>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting)
    L.write_log(s)
    exp.train(setting)

    s = '>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting)
    L.write_log(s)
    exp.test(setting)

    torch.cuda.empty_cache()

