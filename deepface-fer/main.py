# set-up in Terminal:
# pip install opencv-python
# pip install fer
# pip install pandas
# pip install tensorflow
# pip install fer-pytorch
import cv2
import pandas as pd
# from fer import FER
from deepface import DeepFace
import time
from fer_pytorch.fer import FER

# path = "dataset\\train\\"
path = "dataset\\test\\"
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# totalImages = [3995, 436, 4097, 7215, 4965, 4830, 3171]
totalImages = [958, 111, 1024, 1774, 1233, 1247, 831]

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


# anger disgust fear happiness neutral sadness surprise
def apply_fer_pytorch(img):
    global real_count
    fer = FER()
    fer.get_pretrained_model("resnet34")
    # result1 = fer.predict_image(img)
    # print(result1)
    result = fer.predict_image(img, show_top=True)
    # print(result)
    if len(result) > 0:
        # cv2.imshow(str(real_count), img)
        real_count = real_count + 1
        res = list(((result[0])["top_emotion"]).keys())[0]
        if res == "happiness":
            return "happy"
        if res == "sadness":
            return "sad"
        return res
    else:
        return ""


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

                #result = apply_fer_pytorch(img)
                #if result != emotion:
                if apply_fer_pytorch(img) != emotion:
                    wrong = wrong + 1
                #    if result != "":
                #        cv2.imshow("emotion" + str(wrong), img)

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
    count()
    # cv2.waitKey(0)

