import numpy as np
import argparse
import os
import pickle
import scipy
import trimesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, default="/home/federico/Scrivania/FaceFormer/BIWI/elaborated_result")
    #parser.add_argument("--pred_path", type=str, default="/home/federico/Scrivania/ST/Data/predictions/BIWI")
    #parser.add_argument("--gt_path", type=str, default="/home/federico/Scrivania/TH/S2L/vocaset/vertices_npy")
    parser.add_argument("--gt_path", type=str, default="/home/federico/Scrivania/ST/Data/Biwi_6/vertices")
    #parser.add_argument("--gt_path", type=str, default="/home/federico/Scrivania/ST/Data/Multiface/vertices")
    #parser.add_argument("--templates_path", type=str, default="/home/federico/Scrivania/ST/Data/Multiface/templates")
    parser.add_argument("--templates_path", type=str, default="/home/federico/Scrivania/ST/Data/Biwi_6/templates")
    #parser.add_argument("--templates_path", type=str, default="/home/federico/Scrivania/TH/S2L/vocaset/templates.pkl")
    parser.add_argument("--dataset", type=str, default="BIWI")
    args = parser.parse_args()

    if args.dataset == "vocaset":
        with open(args.templates_path, 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')

        lip_mask_voca = scipy.io.loadmat('/home/federico/Scrivania/ST/ScanTalk/FLAME_lips_idx.mat')
        lip_mask_voca = lip_mask_voca['lips_idx'] - 1 
        lip_mask_voca = np.reshape(np.array(lip_mask_voca, dtype=np.int64), (lip_mask_voca.shape[0]))
        lip_mask = lip_mask_voca.tolist()
        
        upper_mask = np.load('/mnt/diskone-second/D2D/Masks/upper_mask.npy').tolist()
        
        nr_vertices = 5023
        
    if args.dataset == "BIWI":
        
        templates = {}
        
        for temp in os.listdir(args.templates_path):
            subject = temp.split(".")[0]
            face_mesh = trimesh.load(os.path.join(args.templates_path, temp), process=False)
            templates[subject] = face_mesh.vertices
        
        upper_mask = np.load('/home/federico/Scrivania/ST/Data/Biwi_6/upper_indices.npy').tolist()
        lip_mask = np.load('/home/federico/Scrivania/ST/Data/Biwi_6/mouth_indices.npy').tolist()
        
        nr_vertices = 3895
    
    if args.dataset == "multiface":
        
        templates = {}
        
        for temp in os.listdir(args.templates_path):
            subject = temp.split(".")[0]
            face_mesh = trimesh.load(os.path.join(args.templates_path, temp), process=False)
            templates[subject] = face_mesh.vertices
        
        upper_mask = np.load('/home/federico/Scrivania/ST/Data/Multiface/upper_indices.npy').tolist()
        lip_mask = np.load('/home/federico/Scrivania/ST/Data/Multiface/mouth_indices.npy').tolist()
        
        nr_vertices = 5471

    cnt = 0
    vertices_gt_all = []
    vertices_pred_all = []
    motion_std_difference = []
    abs_motion_std_difference = []

    mve = 0
    num_seq = 0
    for sentence in os.listdir(args.pred_path):
        if args.dataset == "vocaset":
            subject = sentence.split("s")[0][:-1]
        if args.dataset == "BIWI" or args.dataset == "multiface":
            subject = sentence.split("_")[0]
        
        vertices_gt = np.load(os.path.join(args.gt_path, sentence)).reshape(-1, nr_vertices, 3)

        vertices_pred = np.load(os.path.join(args.pred_path, sentence)).reshape(-1, nr_vertices, 3)

        vertices_pred = vertices_pred[:vertices_gt.shape[0], :, :]
        vertices_gt = vertices_gt[:vertices_pred.shape[0], :, :]

        print(vertices_pred.shape)
        mve += np.linalg.norm(vertices_gt - vertices_pred, axis = 2).mean(axis=1).mean()

        motion_pred = vertices_pred - templates[subject].reshape(1, nr_vertices, 3)
        motion_gt = vertices_gt - templates[subject].reshape(1, nr_vertices, 3)

        cnt += vertices_gt.shape[0]

        vertices_gt_all.extend(list(vertices_gt))
        vertices_pred_all.extend(list(vertices_pred))

        L2_dis_upper = np.array([np.square(motion_gt[:, v, :]) for v in upper_mask])
        L2_dis_upper = np.transpose(L2_dis_upper, (1, 0, 2))
        L2_dis_upper = np.sum(L2_dis_upper, axis=2)
        L2_dis_upper = np.std(L2_dis_upper, axis=0)
        gt_motion_std = np.mean(L2_dis_upper)

        L2_dis_upper = np.array([np.square(motion_pred[:, v, :]) for v in upper_mask])
        L2_dis_upper = np.transpose(L2_dis_upper, (1, 0, 2))
        L2_dis_upper = np.sum(L2_dis_upper, axis=2)
        L2_dis_upper = np.std(L2_dis_upper, axis=0)
        pred_motion_std = np.mean(L2_dis_upper)

        motion_std_difference.append(gt_motion_std - pred_motion_std)
        abs_motion_std_difference.append(np.abs(gt_motion_std - pred_motion_std))
        print(f"{sentence}")
        print('FDD: {:.4e}'.format(motion_std_difference[-1]), 'FDD: {:.4e}'.format(sum(motion_std_difference) / len(motion_std_difference)))

        num_seq += 1

    print('Frame Number: {}'.format(cnt))

    vertices_gt_all = np.array(vertices_gt_all)
    vertices_pred_all = np.array(vertices_pred_all)

    print(vertices_gt_all.shape)

    distances = np.linalg.norm(vertices_gt_all - vertices_pred_all, axis=2)
    mean_distance = np.mean(distances)

    L2_dis_mouth_max = np.array([np.square(vertices_gt_all[:, v, :] - vertices_pred_all[:, v, :]) for v in lip_mask])
    L2_dis_mouth_max = np.transpose(L2_dis_mouth_max, (1, 0, 2))
    L2_dis_mouth_max = np.sum(L2_dis_mouth_max, axis=2)
    L2_dis_mouth_max = np.max(L2_dis_mouth_max, axis=1)

    print('Mean Vertex Error: {:.4e}'.format(mean_distance))
    print('Lip Vertex Error: {:.4e}'.format(np.mean(L2_dis_mouth_max)))
    print('FDD: {:.4e}'.format(sum(motion_std_difference) / len(motion_std_difference)))
    print('ABS FDD: {:.4e}'.format(sum(abs_motion_std_difference) / len(motion_std_difference)))

'''
def compute_diversity():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_subjects", type=str, default="F1 F5 F6 F7 F8 M1 M2 M6")
    parser.add_argument("--pred_path", type=str, default="RUN/BIWI/CodeTalker_s2/result/npy/")
    parser.add_argument("--gt_path", type=str, default="/data/BIWI/vertices_npy/")
    parser.add_argument("--region_path", type=str, default="/data/BIWI/regions/")
    parser.add_argument("--templates_path", type=str, default="/data/BIWI/templates.pkl")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--num_sample", type=str)
    parser.add_argument("--dataset", type=str, default="BIWI")
    args = parser.parse_args()

    train_subject_list = args.train_subjects.split(" ")
    test_subject_list = args.test_subjects.split(" ")


    if args.dataset == "BIWI":
        sentence_list = ["e" + str(i).zfill(2) for i in range(37, 41)]
        nr_vertices = 23370
    else:
        nr_vertices = 6172

        sentence_list = [str(i) for i in range(46, 51)]

    num_seq = 0
    diversity = 0
    for subject in test_subject_list:
        for sentence in sentence_list:

            print(subject, sentence)
            all_pred_seq = []
            for condition in train_subject_list:
                if not os.path.exists(os.path.join(args.pred_path, subject + "_" + sentence + "_condition_" + condition + ".npy")):
                    continue
                vertices_pred = np.load(
                    os.path.join(args.pred_path, subject + "_" + sentence + "_condition_" + condition + ".npy")).reshape(
                    -1,
                    nr_vertices,
                    3)
                all_pred_seq.append(vertices_pred)

            tottal_diff_seq = 0
            n_seq = len(all_pred_seq)
            if n_seq < 2:
                continue
            for i in range(n_seq - 1):
                for j in range(i + 1, n_seq):
                    tottal_diff_seq += np.linalg.norm(all_pred_seq[i] - all_pred_seq[j], axis=2).mean(axis=1).mean()
            tottal_diff_seq /= ((n_seq - 1) * n_seq / 2)
            print(tottal_diff_seq)
            diversity += tottal_diff_seq

            num_seq += 1

    print('Diversity: {:.4e}'.format(diversity / num_seq))
'''

if __name__ == "__main__":
    main()
    #compute_diversity()
