import cv2
import asyncio
import numpy as np
import face_recognition
import os
from datetime import datetime
import telegram

BOT_TOKEN = "7751924712:AAGOzggzZK30ff5gQz4HIz-FHCSBJKPYW5Y"
CHAT_ID = "697396654"

bot = telegram.Bot(token=BOT_TOKEN)
alert_sent = False

async def send_alert_message(name, image_path):
    global alert_sent
    if not alert_sent:
        message = f"{name} kameraya yakalandı!"
        await bot.send_message(chat_id=CHAT_ID, text=message)
        with open(image_path, 'rb') as photo:
            await bot.send_photo(chat_id=CHAT_ID, photo=photo)
        alert_sent = True

data = [
    {"id": 1, "name": "Metin", "age": 24, "sex": "Male", "nationality": "Turkey"},
    {"id": 2, "name": "Ronaldo", "age": 39, "sex": "Male", "nationality": "Portugal"},
    {"id": 3, "name": "Neymar", "age": 32, "sex": "Male", "nationality": "Brasil"},
    {"id": 4, "name": "Balotelli", "age": 34, "sex": "Male", "nationality": "Italy"}
]

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtsString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtsString}')

def display_all_persons_info(img, name):
    y_offset = 30 
    for person in data:
        if person["name"].lower() == name.lower():
            text_name = f"Name: {person['name']}"
            text_age = f"Age: {person['age']}"
            text_sex = f"Sex: {person['sex']}"
            text_ntn = f"Nationality: {person['nationality']}"

            (w_name, h_name), _ = cv2.getTextSize(text_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            (w_age, h_age), _ = cv2.getTextSize(text_age, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            (w_sex, h_sex), _ = cv2.getTextSize(text_sex, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            (w_ntn, h_ntn), _ = cv2.getTextSize(text_ntn, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            cv2.rectangle(img, (10, y_offset - 5), (10 + max(w_name, w_age, w_sex, w_ntn), y_offset + h_name + h_age + 10 + h_sex + 30 + h_ntn + 30), (0, 255, 0), -1)

            cv2.putText(img, text_name, (10, y_offset + h_name), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, text_age, (10, y_offset + h_name + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, text_sex, (10, y_offset + h_name + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, text_ntn, (10, y_offset + h_name + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

async def main():
    global alert_sent
    encodeListKnow = findEncodings(images)

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        if len(facesCurFrame) == 0:
            cv2.putText(img, "No face found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()

                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                markAttendance(name)
                display_all_persons_info(img, name)
                image_path = f"detected_{name}.jpg"
                cv2.imwrite(image_path, img)
                await send_alert_message(name, image_path)

        if cv2.waitKey(1) == 27:
            break

        cv2.imshow('Yuz Tanima Sistemi', img)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
