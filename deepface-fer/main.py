# set-up in Terminal:
# pip install opencv-python
# pip install fer
# pip install pandas
# pip install tensorflow
# pip install fer-pytorch
import cv2
# from fer import FER
from deepface import DeepFace
import time
from fer_pytorch.fer import FER
import numpy as np

from CNN.CNNModel import apply_CNN
import copy

# path = "dataset\\train\\"
path = "dataset\\test\\"
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# totalImages = [3995, 436, 4097, 7215, 4965, 4830, 3171]
totalImages = [958, 111, 1024, 1774, 1233, 1247, 831]

# # results from sprint 2:
# accuracyDeepface = [42.80, 44.14, 41.41, 76.38, 56.20, 41.78, 70.40]
# accuracyPyTorchFER = [31.81, 27.27, 6.04, 86.79, 90.53, 24.33, 66.77]
# accuracyCNN = [35.07, 24.32, 31.54, 78.07, 43.14, 44.34, 64.74]

# improved weights - last sprint
accuracyDeepface = [42.8, 94.14, 41.41, 76.38, 56.2, 41.78, 70.4]
accuracyPyTorchFER = [31.81, 27.27, 6.04, 136.79, 140.53, 24.33, 66.77]
accuracyCNN = [35.07, 24.32, 31.54, 78.07, 43.14, 94.34, 64.74]

# here we calculate the weights for the voting algorithm
sum_all = np.array(accuracyDeepface) + np.array(accuracyPyTorchFER) + np.array(accuracyCNN)
sum_withoutFER = np.array(accuracyDeepface) + np.array(accuracyCNN)
weightsDeepface = np.array(accuracyDeepface) / np.array(sum_all)
weightsPyTorchFER = np.array(accuracyPyTorchFER) / np.array(sum_all)
weightsCNN = np.array(accuracyCNN) / np.array(sum_all)
weightsDeepface_withoutFER = np.array(accuracyDeepface) / np.array(sum_withoutFER)
weightsCNN_withoutFER = np.array(accuracyCNN) / np.array(sum_withoutFER)

print("weightsDeepface")
print(weightsDeepface)
print("weightsPyTorchFER")
print(weightsPyTorchFER)
print("weightsCNN")
print(weightsCNN)

# because deepface results are percentages, 100x times higher than fer & cnn
weightsDeepface *= 0.01
weightsDeepface_withoutFER *= 0.01

# example for happy:
# deepface: 72.60763049125671 --> 0.72      accuracy: 76.38
# fer: 0.80355483                           accuracy: 86.79
# CNN: 3.0142978e-01 === 0.30142978         accuracy: 77.67
# sum: 76.38 + 86.79 + 77.67 --> 240.84
# result: (0.72 * 76.38 + 0.80 * 86.79 + 0.301 * 77.67) / 240.84 --> (55 + 69.43 + 23.37)  / 240.84 --> 0.613
# we got: [0.09912332 0.07628439 0.16035434  --> 0.61658874 <-- 0.16090182 0.14361914 0.14081906] correct

real_count = 0


# angry disgust fear happy neutral sad surprise
def apply_fer(img):
    detector = FER()
    # Save output in result variable
    arr = [[0, 0, 48, 48]]
    # result = detector.detect_emotions(img, arr)
    # Output image's information
    # print(result)
    # print(detector.top_emotion(img)[0])
    return detector.top_emotion(img, arr)[0]

    # emotions = result[0]["emotions"]
    # for index, (emotion_name, score) in enumerate(emotions.items()):
    #     emotion_score = "{}: {}".format(emotion_name, "{:.2f}".format(score))
    #     print(emotion_score)


# angry disgust fear happy neutral sad surprise
def apply_deepface(img_path):
    result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
    # print(result)
    return result["dominant_emotion"]


def get_results_deepface(img_path):
    result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
    result = result["emotion"]
    result = list(result.values())
    return result


# anger disgust fear happiness neutral sadness surprise
def apply_fer_pytorch(img):
    global real_count
    fer = FER()
    fer.get_pretrained_model("resnet34")
    # result1 = fer.predict_image(img)
    # print(result1)
    result = fer.predict_image(img, show_top=True)
    # print(result) # [{'box': [7.709435, 1.4501405, 41.738373, 48.494156], 'top_emotion': {'neutral': 0.5612427}}]
    if len(result) > 0:
        # cv2.imshow(str(real_count), img)
        real_count = real_count + 1
        res = list(((result[0])["top_emotion"]).keys())[0]
        if res == "happiness":
            return "happy"
        if res == "sadness":
            return "sad"
        if res == "anger":
            return "angry"
        return res
    else:
        return ""


def get_results_fer(img):
    fer = FER()
    fer.get_pretrained_model("resnet34")
    result = fer.predict_image(img)

    if len(result) > 0:
        result = (result[0])["emotions"]
        result = dict(sorted(result.items()))
        result = list(result.values())
        return result
    else:
        return ""


def get_results_CNN(img):
    result = apply_CNN(img)
    result = result[0]
    return result


def get_voting_based(img_path):
    img = cv2.imread(img_path)
    result_deepface = get_results_deepface(img_path)
    result_fer = get_results_fer(img)
    result_CNN = get_results_CNN(img)

    result = []
    if result_fer == "":
        result = np.array(result_deepface) * weightsDeepface_withoutFER + np.array(result_CNN) * weightsCNN_withoutFER
    else:
        result = np.array(result_deepface) * weightsDeepface + np.array(result_fer) * weightsPyTorchFER + np.array(result_CNN) * weightsCNN
    # print(result)

    # sort the emotions in descending order
    result = dict(zip(emotions, result))
    result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    return result


def apply_voting_based_top(img_path):
    result = get_voting_based(img_path)
    return list(result.keys())[0]


d = []
def createMatrix(i,  result):
    global d
    d.append(np.array(result))


result_err = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
def count():
    global d
    global real_count
    global result_err
    with open('filename.txt', 'w') as f:
        for emotionIndex in range(0, 7):
            emotion = emotions[emotionIndex]
            image_count = totalImages[emotionIndex]
            real_count = 0
            wrong = 0
            total_time = 0
            d = []
            for image_index in range(0, image_count):
                img_path = path + emotion + "\\im" + str(image_index) + ".png"
                img = cv2.imread(img_path)
                # start = time.perf_counter()

                # if apply_fer(img) != emotion:
                #     wrong = wrong + 1
                # real_count = image_count

                # if apply_deepface(img_path) != emotion:
                #     wrong = wrong + 1
                # real_count = image_count

                # #result = apply_fer_pytorch(img)
                # #if result != emotion:
                # if apply_fer_pytorch(img) != emotion:
                #     wrong = wrong + 1
                # #    if result != "":
                # #        cv2.imshow("emotion" + str(wrong), img)

                # if apply_voting_based_top(img_path) != emotion:
                #     wrong = wrong + 1
                # real_count = image_count

                # result = get_results_deepface(c)
                result = get_results_CNN(img)
                # result = get_results_fer(img)
                # if result == "":
                #     result = result_err
                createMatrix(image_index, result)
                #print(d)

                # stop = time.perf_counter()
                # total_time = total_time + stop - start

            # print("d este")
            # print(d)

            filename = 'CNN_' + emotion + '.txt'
            np.savetxt(filename, d, fmt='%f')



            # wrong = wrong - image_count + real_count
            # print(str(wrong) + " / " + str(real_count) +
            #       ", percentage correct: " + str((real_count - wrong) / real_count * 100) +
            #       "%, average time: " + str(total_time / real_count), file=f)
            # print(str(wrong) + " / " + str(real_count) +
            #       ", percentage correct: " + str((real_count - wrong) / real_count * 100) +
            #       "%, average time: " + str(total_time / real_count))


def update_weights():
    global weightsDeepface
    global weightsCNN
    global weightsPyTorchFER
    global weightsCNN_withoutFER
    global weightsDeepface_withoutFER

    # # results from sprint 2:
    # accuracyDeepface =   [42.80, 44.14, 41.41, 76.38, 56.20, 41.78, 70.40]
    # accuracyPyTorchFER = [31.81, 27.27, 6.04, 86.79, 90.53, 24.33, 66.77]
    # accuracyCNN =        [36.32, 24.32, 31.83, 77.67, 42.49, 43.70, 65.34]

    # 46.161955 40.501044 42.342342 40.429688 79.988726 39.253852 19.246191 61.371841
    # accuracyDeepface = [90, 90, 90, 90, 90, 1, 90]
    # accuracyPyTorchFER = [31, 27, 0, 86, 97, 1, 66]
    # accuracyCNN = [36, 24, 31, 77, 1, 90, 65]

    # here we calculate the weights for the voting algorithm
    sum_all = np.array(accuracyDeepface) + np.array(accuracyPyTorchFER) + np.array(accuracyCNN)
    sum_withoutFER = np.array(accuracyDeepface) + np.array(accuracyCNN)
    weightsDeepface = np.array(accuracyDeepface) / np.array(sum_all)
    weightsPyTorchFER = np.array(accuracyPyTorchFER) / np.array(sum_all)
    weightsCNN = np.array(accuracyCNN) / np.array(sum_all)
    weightsDeepface_withoutFER = np.array(accuracyDeepface) / np.array(sum_withoutFER)
    weightsCNN_withoutFER = np.array(accuracyCNN) / np.array(sum_withoutFER)

    # because deepface results are percentages, 100x times higher than fer & cnn
    weightsDeepface *= 0.01
    weightsDeepface_withoutFER *= 0.01

percentages = []
def add_percentage(percentage):
    global percentages
    percentage.insert(0, np.sum(percentage) / 7.0)
    percentages.append(percentage)


def permute(list, s):
   if list == 1:
      return s
   else:
      return [
         y + x
         for y in permute(1, s)
         for x in permute(list - 1, s)
      ]

def count2():
    global result_err
    global percentages
    global accuracyDeepface
    global accuracyCNN
    global accuracyPyTorchFER

    add_percentage([42.8, 44.14, 41.41, 86.79, 90.53, 43.7, 70.4])
    add_percentage(accuracyDeepface)
    add_percentage(accuracyPyTorchFER)
    add_percentage(accuracyCNN)
    add_percentage([0, 0, 0, 0, 0, 0, 0])

    accuracyDeepface_perm = np.loadtxt('accuracyDeepface_perm.txt', dtype=float)
    accuracyPyTorchFER_perm = np.loadtxt('accuracyPyTorchFER_perm.txt', dtype=float)
    accuracyCNN_perm = np.loadtxt('accuracyCNN_perm.txt', dtype=float)
    iterations_count = len(accuracyDeepface_perm)

    for iteration in range (0, iterations_count):
        print(str(iteration) + '/' + str(iterations_count))
        # recalculate the voting algorithm parameters
        accuracyDeepface = accuracyDeepface_perm[iteration]
        accuracyPyTorchFER = accuracyPyTorchFER_perm[iteration]
        accuracyCNN = accuracyCNN_perm[iteration]
        update_weights()

        # use the voting algorithm on the dataset and get the percentage
        percentage = []
        for emotionIndex in range(0, 7):
            emotion = emotions[emotionIndex]
            image_count = totalImages[emotionIndex]
            wrong = 0
            real_count = 0
            # load the results for all the images
            values_deepface = np.loadtxt('deepface_' + emotion + '.txt', dtype=float)
            values_fer = np.loadtxt('fer_' + emotion + '.txt', dtype=float)
            values_CNN = np.loadtxt('cnn_' + emotion + '.txt', dtype=float)
            # for each image, find if the result is the correct one
            for image_index in range(0, image_count):
                result = []
                result_deepface = values_deepface[image_index]
                result_fer = values_fer[image_index]
                result_CNN = values_CNN[image_index]
                if (result_fer == result_err).all():
                    result = np.array(result_deepface) * weightsDeepface_withoutFER + np.array(
                        result_CNN) * weightsCNN_withoutFER
                else:
                    result = np.array(result_deepface) * weightsDeepface + np.array(
                        result_fer) * weightsPyTorchFER + np.array(result_CNN) * weightsCNN
                # print(result)

                # sort the emotions in descending order
                result = dict(zip(emotions, result))
                result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
                result = list(result.keys())[0]
                if result != emotion:
                    wrong = wrong + 1
                real_count = real_count + 1
            # append the percentage for this emotion
            percentage.append((real_count - wrong) / real_count * 100)

        # here we have the result for the whole dataset, for the current iteration
        # we add it in a big array
        add_percentage(percentage)

    # here we print the array with all the results of all iterations in a file
    filename = 'percentages.txt'
    np.savetxt(filename, percentages, fmt='%f')


def generate_accuracies():
    print('hello')
    permutations = permute(7, [[0], [1]])
    filename = 'permutations.txt'
    np.savetxt(filename, permutations, fmt='%d')

    values_permutations = np.loadtxt('permutations.txt', dtype=int)
    values_count = len(values_permutations)

    max_indices = []
    min_indices = []
    a = [1, 2, 30]
    b = [4, 5, 6]
    c = [7, 2, 3]

    for iterator in range(0, 7):
        array = np.array([accuracyDeepface[iterator], accuracyPyTorchFER[iterator], accuracyCNN[iterator]])
        max_indices.append(array.argmax())
        min_indices.append(array.argmin())

    print(max_indices)
    print(min_indices)

    addValue = 50.0
    subValue = 50.0

    accuracyDeepface_perm = []
    accuracyPyTorchFER_perm = []
    accuracyCNN_perm = []

    add = [ True, False, True]
    sub = [ False, True, True]
    for caseIterator in range(0, 3):
        for iterator in range(0, values_count):
            #print(values_permutations[iterator])
            acc_deep = np.array(accuracyDeepface)
            acc_fer = np.array(accuracyPyTorchFER)
            acc_cnn = np.array(accuracyCNN)
            for value_it in range(0, 7):

                if add[caseIterator]:
                    if values_permutations[iterator][value_it] == 1:
                        max_pos = max_indices[value_it]
                        if max_pos == 0:
                            acc_deep[value_it] += addValue
                        elif max_pos == 1:
                            acc_fer[value_it] += addValue
                        else:
                            acc_cnn[value_it] += addValue

                if sub[caseIterator]:
                    if values_permutations[iterator][value_it] == 1:
                        min_pos = min_indices[value_it]
                        if min_pos == 0:
                            acc_deep[value_it] -= subValue
                            if acc_deep[value_it] <= 0: acc_deep[value_it] = 0.01
                        elif min_pos == 1:
                            acc_fer[value_it] -= subValue
                            if acc_fer[value_it] <= 0: acc_fer[value_it] = 0.01
                        else:
                            acc_cnn[value_it] -= subValue
                            if acc_cnn[value_it] <= 0: acc_cnn[value_it] = 0.01

            accuracyDeepface_perm.append(acc_deep)
            accuracyPyTorchFER_perm.append(acc_fer)
            accuracyCNN_perm.append(acc_cnn)

    np.savetxt('accuracyDeepface_perm.txt', accuracyDeepface_perm, fmt='%f')
    np.savetxt('accuracyPyTorchFER_perm.txt', accuracyPyTorchFER_perm, fmt='%f')
    np.savetxt('accuracyCNN_perm.txt', accuracyCNN_perm, fmt='%f')



def get_maximum():
    percentages = np.loadtxt('percentages.txt', dtype=float)
    percentages = np.array(percentages[5:])
    # a = percentages.max(axis = 0)
    b = percentages.argmax(axis = 0)
    # print(a)
    print(b)
    print(percentages[b[0]])
    accuracyDeepface_perm = np.loadtxt('accuracyDeepface_perm.txt', dtype=float)
    accuracyPyTorchFER_perm = np.loadtxt('accuracyPyTorchFER_perm.txt', dtype=float)
    accuracyCNN_perm = np.loadtxt('accuracyCNN_perm.txt', dtype=float)
    print(accuracyDeepface_perm[b[0]])
    print(accuracyPyTorchFER_perm[b[0]])
    print(accuracyCNN_perm[b[0]])


def get_accuracy():
    p = []
    for emotionIndex in range(0, 7):
        emotion = emotions[emotionIndex]
        image_count = totalImages[emotionIndex]
        filename = 'CNN_' + emotion + '.txt'
        values = np.loadtxt(filename, dtype=float)
        wrong = 0
        for imageIterator in range(0, image_count):
            max = values[imageIterator].argmax()
            if max != emotionIndex:
                wrong = wrong + 1
        p.append((image_count - wrong) / image_count * 100)
    print(p)

if __name__ == '__main__':
    img_path = path + "happy\\im3.png"
    img = cv2.imread(img_path)
    result = get_voting_based(img_path)
    print(result)
    print(list(result.keys())[0])
    # factor = 1.0 / sum(result.values())
    # for k in result:
    #    result[k] = result[k] * factor
    # print(result)

    #count()
    # get_accuracy()
    # count2()

    # generate_accuracies()
    # count2()
    # get_maximum()


    # cv2.waitKey(0)


