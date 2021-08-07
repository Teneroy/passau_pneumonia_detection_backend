from emd import *
from os import listdir


def research():
    #img_a = get_img('emd_research_data/xray5.png', norm_exposure=True)
   # hist_a = get_histogram(img_a)

    rightList = []
    wrongList = []
    tempList = []

    resultList = []

    thr = 0.0009
    for i in range(29):
        correct_number = 0
        correct_wrong = 0
        correct_right = 0
        total_number = 0
        print("THRESHOLD = " + str(thr))
        print("WRONG DATASET")
        for pathW in listdir('./emd_research_data/wrong/'):
            img_test = get_img('./emd_research_data/wrong/' + pathW, norm_exposure=True)
            is_xray = get_distance(img_test) <= thr
            if not is_xray:
                correct_number += 1
                correct_wrong += 1
            print(pathW + " " + str(is_xray))
            total_number += 1
        print("\n RIGHT DATASET")
        for pathR in listdir('./emd_research_data/right/'):
            img_test = get_img('./emd_research_data/right/' + pathR, norm_exposure=True)
            is_xray = get_distance(img_test) <= thr
            if is_xray:
                correct_number += 1
                correct_right += 1
            print(pathR + " " + str(is_xray))
            total_number += 1
        thr += 0.00005
        print("CORRECT: " + str(correct_number) + " , TOTAL: " + str(total_number))
        resultList.append((thr, correct_number, ("w", correct_wrong), ("r", correct_right)))
        print("____________________")

    print(resultList)


research()

# print(emd.emd_comparison('r1.png'))
# print(emd.emd_comparison('w1.png'))

# r = [0.002639924778657801, 0.0025472151927458933, 0.002450862876799425, 0.0023673694867354174, 0.0022952713809170568, 0.0022757032534459253, 0.0022085745597762805, 0.0021473123477055477, 0.0020995910351093, 0.002047558860665948, 0.0020013561955204715, 0.0019964782091287468, 0.0019534073710757794]
# w = [0.0025343528160682092, 0.0018815214817340558, 0.0015999445548424353, 0.0019087562194237341, 0.0021738602564885068, 0.0028353073658087314, 0.0028774266714578145, 0.0029167189047886776, 0.0028610555534688835, 0.002900103422311636, 0.0027254561444262524, 0.002643145047701322, 0.002556597692726632, 0.0027658775612548155, 0.002651522098443447, 0.0027167023374484135]
#
# print("Max r: ", max(r))
# print("Min r: ", min(r))
# print("Max w: ", max(w))
# print("Min w: ", min(w))
#
# print("AVG RIGHT: ")
# print(sum(r) / len(r))
# print(sum(w) / len(w))
#
#
# print("r sorted")
# print(sorted(r))
# print("w sorted")
# print(sorted(w))