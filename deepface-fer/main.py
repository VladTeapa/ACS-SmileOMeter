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

# path = "dataset\\train\\"
path = "dataset\\test\\"
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# totalImages = [3995, 436, 4097, 7215, 4965, 4830, 3171]
totalImages = [958, 111, 1024, 1774, 1233, 1247, 831]

# results from sprint 2:
accuracyDeepface = [42.80, 44.14, 41.41, 76.38, 56.20, 41.78, 70.40]
accuracyPyTorchFER = [31.81, 27.27, 6.04, 86.79, 90.53, 24.33, 66.77]
accuracyCNN = [36.32, 24.32, 31.83, 77.67, 42.49, 43.70, 65.34]

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

    # print("result_deepface")
    # print(result_deepface)
    # print("result_fer")
    # print(result_fer)
    # print("result_CNN")
    # print(result_CNN)
    result_CNN = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

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

def count():
    global real_count
    with open('filename.txt', 'w') as f:
        for emotionIndex in range(0, 7):
            emotion = emotions[emotionIndex]
            image_count = totalImages[emotionIndex]
            real_count = 0
            wrong = 0
            total_time = 0
            for image_index in range(0, image_count):
                img_path = path + emotion + "\\im" + str(image_index) + ".png"
                img = cv2.imread(img_path)
                start = time.perf_counter()

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

                if apply_voting_based_top(img_path) != emotion:
                    wrong = wrong + 1
                real_count = image_count

                stop = time.perf_counter()
                total_time = total_time + stop - start

            wrong = wrong - image_count + real_count
            print(str(wrong) + " / " + str(real_count) +
                  ", percentage correct: " + str((real_count - wrong) / real_count * 100) +
                  "%, average time: " + str(total_time / real_count), file=f)
            print(str(wrong) + " / " + str(real_count) +
                  ", percentage correct: " + str((real_count - wrong) / real_count * 100) +
                  "%, average time: " + str(total_time / real_count))

if __name__ == '__main__':
    img_path = path + "happy\\im0.png"
    img = cv2.imread(img_path)
    result = get_voting_based(img_path)
    print(result)
    print(list(result.keys())[0])

    # count()
    # cv2.waitKey(0)


