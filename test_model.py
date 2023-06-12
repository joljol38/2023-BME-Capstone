import os
import torch
import random
import pickle
import argparse
import numpy as np
import torch.nn as nn
import sys
import time
from math import sqrt
import torch.utils.data
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from network import ConvNCF

from sklearn.metrics.pairwise import cosine_similarity

def read_raw_data(rawdata_dir):
    gii = open(rawdata_dir + '/' + 'SMD_combined.pkl', 'rb')
    drug_Tfeature_one = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'SMD_database.pkl', 'rb')
    drug_Tfeature_two = pickle.load(gii)
    gii.close()


    gii = open(rawdata_dir + '/' + 'SMD_experimental.pkl', 'rb')
    drug_Tfeature_three = pickle.load(gii)
    gii.close()


    gii = open(rawdata_dir + '/' + 'SMD_similarity.pkl', 'rb')
    drug_Tfeature_four = pickle.load(gii)
    gii.close()


    gii = open(rawdata_dir + '/' + 'SMD_textmining.pkl', 'rb')
    drug_Tfeature_five = pickle.load(gii)
    gii.close()


    gii = open(rawdata_dir + '/' + 'side_effect_semantic.pkl', 'rb')
    effect_side_semantic = pickle.load(gii)
    gii.close()


    # gii = open(rawdata_dir + '/' + 'drug_mol.pkl', 'rb')
    # Drug_word2vec = pickle.load(gii)
    # gii.close()
    # Drug_word_sim = cosine_similarity(Drug_word2vec)


    gii = open(rawdata_dir + '/' + 'new_glove_wordEmbedding.pkl', 'rb')
    glove_word = pickle.load(gii)
    gii.close()
    side_glove_sim = cosine_similarity(glove_word)

    gii = open(rawdata_dir + '/' + 'drug_gene_target.pkl', 'rb')
    drug_target = pickle.load(gii)
    gii.close()
    drug_target_sim = cosine_similarity(drug_target)


    gii = open(rawdata_dir + '/' + 'drug_structure_similarity.pkl', 'rb')
    drug_f_sim = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'drug_side_freq2.pkl', 'rb')
    drug_side = pickle.load(gii)
    gii.close()

    drug_side_sim = cosine_similarity(drug_side)

    drug_side_label = np.zeros((drug_side.shape[0], drug_side.shape[1]))
    for i in range(drug_side.shape[0]):
        for j in range(drug_side.shape[1]):
            if drug_side[i, j] > 0:
                drug_side_label[i, j] = 1
    drug_side_label_sim = cosine_similarity(drug_side_label)

    drug_features, side_features = [], []
    drug_features.append(drug_Tfeature_one)
    drug_features.append(drug_Tfeature_two)
    drug_features.append(drug_Tfeature_three)
    drug_features.append(drug_Tfeature_four)
    drug_features.append(drug_Tfeature_five)
    drug_features.append(drug_target_sim)
    drug_features.append(drug_f_sim)
    drug_features.append(drug_side_sim)
    drug_features.append(drug_side_label_sim)

    side_drug_sim = cosine_similarity(drug_side.T)
    side_drug_label_sim = cosine_similarity(drug_side_label.T)


    side_features.append(effect_side_semantic)
    side_features.append(side_glove_sim)
    side_features.append(side_drug_sim)
    side_features.append(side_drug_label_sim)


    return drug_features, side_features

def fold_files(args):
    rawdata_dir = args.rawpath

    drug_features, side_features = read_raw_data(rawdata_dir)

    drug_features_matrix = drug_features[0]
    for i in range(1, len(drug_features)):
        drug_features_matrix = np.hstack((drug_features_matrix, drug_features[i]))

    side_features_matrix = side_features[0]
    for i in range(1, len(side_features)):
        side_features_matrix = np.hstack((side_features_matrix, side_features[i]))

    two_cell = []
    for i in range(drug_features_matrix.shape[0]):
        for j in range(side_features_matrix.shape[0]):
            two_cell.append([i, j])

    two_cell = np.array(two_cell)

    drug_test = drug_features_matrix[two_cell[:, 0]]
    side_test = side_features_matrix[two_cell[:, 1]]

    return drug_test, side_test, two_cell

def test_data(args):
    drug_test, side_test, two_cell = fold_files(args)
    print("drug test: ", drug_test)
    print("drug test size: ", drug_test.shape)
    print("side_test: ", side_test)
    print("side test size: ", side_test.shape)
    print("two_cell: ", two_cell)
    print("two_cell size: ", two_cell.shape)
    testset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_test), torch.FloatTensor(side_test))
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    torch.backends.cudnn.benchmark = True

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    model = ConvNCF(4779, 2620, args.embed_dim, args.batch_size).to(device)
    model_file = args.rawpath + '/' + 'my_cap_model22.dat'
    print(model_file)
    checkpoint = torch.load(model_file, map_location = device)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['model'])
    model.eval()
    pred1 = []
    pred2 = []
    for test_drug, test_side in _test:
        scores_one, scores_two = model(test_drug, test_side, device)
        print("test_drug: ", test_drug)
        print("test_side: ", test_side)
        pred1.append(list(scores_one.data.cpu().numpy()))
        pred2.append(list(scores_two.data.cpu().numpy()))
    pred1 = np.array(sum(pred1, []), dtype=np.float32)
    pred2 = np.array(sum(pred2, []), dtype=np.float32)
    print("pred1: ", pred1)
    print("len pred1: ", len(pred1))
    print("pred2: ", pred2)
    print("len pred2: ", len(pred2))


    print('Output_data')
    output = []
    output.append(['drug_id', 'side_effect_id', 'Sample_association_score', 'Sample_frequency_score'])
    for i in range(pred1.shape[0]):
        # if pred1[i] < 0.5:
        #     pred2[i] = 0
        output.append([str(two_cell[i][0]), str(two_cell[i][1]), str(pred1[i]), str(pred2[i])])

    t = ''
    with open('Prediction3.tsv', 'w') as q:
        for i in output:
            for e in range(len(output[0])):
                t = t + str(i[e]) + '\t'
            q.write(t.strip('\t'))
            q.write('\n')
            t = ''

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='FLOAT', help='learning rate')
    parser.add_argument('--embed_dim', type=int, default=128, metavar='N', help='embedding dimension')
    parser.add_argument('--weight_decay', type=float, default=0.00001, metavar='FLOAT', help='weight decay')
    parser.add_argument('--N', type=int, default=30000, metavar='N', help='L0 parameter')
    parser.add_argument('--droprate', type=float, default=0.5, metavar='FLOAT', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='N', help='input batch size for testing')
    parser.add_argument('--dataset', type=str, default='hh', metavar='STRING', help='dataset')
    parser.add_argument('--rawpath', type=str, default='/content/drive/MyDrive/캡스톤/model/data', metavar='STRING', help='rawpath')
    # parser.add_argument('--model_file', type=str, default='/content/drive/MyDrive/캡스톤/model/data/sd_pred_model.dat', metavar='STRING', help='path to model file')
    args = parser.parse_args(args=[])

    print('Dataset: ' + args.dataset)

    print('-------------------- Hyperparams --------------------')
    print('N: ' + str(args.N))
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('learning rate: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))
    test_data(args)

if __name__ == "__main__":
    main()

