import pymeshlab
import os

ids = ['FaceTalk_170725_00137_TA', 'FaceTalk_170728_03272_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170809_00138_TA',
       'FaceTalk_170811_03274_TA', 'FaceTalk_170811_03275_TA', 'FaceTalk_170904_00128_TA', 'FaceTalk_170904_03276_TA',
       'FaceTalk_170908_03277_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170915_00223_TA']

sentences = [f'sentence{i:02}' for i in range(1,41)]
print(sentences)

base_path = "../datasets/VOCA"
output_path = "../datasets/VOCA_remeshed"

os.makedirs(output_path, exist_ok=True)

for id in ids:
       for sentence in sentences:
              i = 1
              length_sentence = len(os.listdir(base_path + "/" + id + "/" + sentence))
              while i <= length_sentence:
                     if not os.path.exists(output_path + "/" + id + "/" + sentence + "/" + str(sentence) + f'.{i:06}' + ".ply") and os.path.exists(base_path + "/" + id + "/" + sentence + "/" + str(sentence) + f'.{i:06}' + ".ply"):
                            # create a new MeshSet
                            ms = pymeshlab.MeshSet()
                            ms.load_new_mesh(base_path + "/" + id + "/" + sentence + "/" + str(sentence) + f'.{i:06}' + ".ply")
                            #ms.load_filter_script('./sub_simp.mlx')
                            ms.apply_filter('meshing_surface_subdivision_butterfly')
                            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=5000)

                            if not os.path.exists(output_path + "/" + id):
                                   os.mkdir(output_path + "/" + id)
                            if not os.path.exists(output_path + "/" + id + "/" + sentence):
                                   os.mkdir(output_path + "/" + id + "/" + sentence)

                            ms.save_current_mesh(
                                   output_path + "/" + id + "/" + sentence + "/" + str(sentence) + f'.{i:06}' + ".ply")
                     i += 1