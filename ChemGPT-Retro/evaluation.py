import os, sys
from rdkit.Chem import AllChem
from rdkit import Chem
from multiprocessing import Pool
from optparse import OptionParser

cwd = os.getcwd()
parser = OptionParser()
parser.add_option("-o", "--output_file", dest="output_file", default='test_result', help='模型输出的预测结果')
parser.add_option("-t", "--test_file", dest="test_file", default = 'groundtruth_path', help='target标签')
parser.add_option("-c", "--num_cores", dest="num_cores", default=10)
parser.add_option("-n", "--top_n", dest="top_n", default=10)
opts, args = parser.parse_args()

num_cores = int(opts.num_cores)
top_n = int(opts.top_n)

def convert_cano(smi):
    try:
        smi = smi.replace('|', '.')  # 不同数据集需要注意这里
        mol = AllChem.MolFromSmiles(smi)
        smiles = Chem.MolToSmiles(mol)
    except:
        smiles = '####'
    return smiles

with open(opts.output_file, 'r') as f:
    pred_targets = f.readlines()  # 读入的文件每一行为10个预测结果，每个预测结果用'\t'符号隔开

with open(opts.test_file, 'r') as f:
    test_targets_list = f.readlines()

if len(pred_targets) != len(test_targets_list):
    raise ValueError('source & target 长度不一致')

pred_targets_beam_10_list = [line.strip().split('\t') for line in pred_targets]  # (5004,10)

num_rxn = len(test_targets_list)
# convert_cano: smile->mol->smile
test_targets_strip_list = [convert_cano(line.replace(' ', '').strip()) for line in test_targets_list]

def smi_valid_eval(ix):
    invalid_smiles = 0
    for j in range(top_n):
        #print('debug1:',len(pred_targets_beam_10_list[0]))
        output_pred_strip = pred_targets_beam_10_list[ix][j].replace(' ', '').strip()
        mol = AllChem.MolFromSmiles(output_pred_strip)
        if mol:
            pass
        else:
            invalid_smiles += 1
    return invalid_smiles

def pred_topn_eval(ix):
    pred_true = 0
    for j in range(top_n):
        output_pred_split_list = pred_targets_beam_10_list[ix][j].replace(' ', '').strip()
        test_targets_split_list = test_targets_strip_list[ix]
        # print(convert_cano(output_pred_split_list), convert_cano(test_targets_split_list))
        if convert_cano(output_pred_split_list) == convert_cano(test_targets_split_list):
            pred_true += 1
            break
        else:
            continue
    return pred_true

if __name__ == "__main__":
    # calculate invalid SMILES rate
    pool = Pool(num_cores)
    invalid_smiles = pool.map(smi_valid_eval, range(num_rxn), chunksize=1)
    invalid_smiles_total = sum(invalid_smiles)
    # calculate predicted accuracy
    pool = Pool(num_cores)
    pred_true = pool.map(pred_topn_eval, range(num_rxn), chunksize=1)
    pred_true_total = sum(pred_true)
    pool.close()
    pool.join()

    # 单线程调试
    # invalid_smiles_total = 0
    # for i in range(num_rxn):
    #     invalid_smiles = smi_valid_eval(i)
    #     invalid_smiles_total += invalid_smiles
    
    # pred_true_total = 0
    # for i in range(num_rxn):
    #     pred_true = pred_topn_eval(i)
    #     pred_true_total += pred_true
    print("Number of invalid SMILES: {}".format(invalid_smiles_total))
    print("Number of SMILES candidates: {}".format(num_rxn * top_n))
    print("Invalid SMILES rate: {0:.3f}".format(invalid_smiles_total / (num_rxn * top_n)))
    print("Number of matched examples: {}".format((pred_true_total)))
    print("Top-{}".format(top_n) +" accuracy: {0:.3f}".format(pred_true_total / num_rxn))

