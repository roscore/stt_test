#!/usr/bin/env python3
import roslib
import rospkg
import rospy
import sys

import io
import os
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

from std_msgs.msg import Int64
from std_msgs.msg import Int32
from std_msgs.msg import Bool

from able_obstacle_msgs.msg import Obstacle

from openpyxl import load_workbook

import time
import random

class AbleStt():
    def __init__(self):
        self.phrase_time = None
        self.source = None
        self.last_sample = bytes()
        self.data_queue = Queue()
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = 1000
        self.recorder.dynamic_energy_threshold = False
        self.check = False  # Initialize self.check
        self.speaker_active = False
        self.action = False

        model = "base"  # 모델 선택 (현재는 base 모델)
        mic_device_path = 'C920' # 사용할 마이크의 경로

        # /dev/logitech_mic 장치를 찾아서 카메라 인덱스 선택
        mic_device_index = None
                
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_device_path in name:
                mic_device_index = index
                break

        os.system('cls' if os.name=='nt' else 'clear')
        print("모델을 로드하는 중 입니다...")

        self.audio_model = whisper.load_model(model)  # model 넣는곳.
        self.record_timeout = 2
        self.phrase_timeout = 3
        self.temp_file = NamedTemporaryFile().name
        self.transcription = ['']

        self.test_msg = None
        self.stt_msg = None
        self.random = 0

        os.system('cls' if os.name=='nt' else 'clear')
        
        if mic_device_index is not None:
            print("Mic device found: " + mic_device_path)
            self.source = sr.Microphone(sample_rate=16000, device_index=mic_device_index)
        else:
            print("Mic device not found: " + mic_device_path)

        dummy_data = 'test.wav'
        # Prevent real transcription by setting self.speaker_active to True
        self.speaker_active = True
        result = self.audio_model.transcribe(dummy_data, language="ko", fp16=torch.cuda.is_available())
        # Reset self.speaker_active to its initial value
        print("Model Prehead Finished")
        self.speaker_active = False

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)  # noise를 조금 줄여주는곳

        def record_callback(_, audio:sr.AudioData) -> None:
            if self.action and not self.speaker_active:  # STT 모드 ON 이고, 스피커가 사용 중이지 않을 때 
                if self.obstacle.mode == "human" or self.obstacle.mode == "many_people":  # 에이블 앞에 사람이 있다면
                    if self.check:  #  에이블이 모션을 취하지 않고 있다면
                        data = audio.get_raw_data()
                        self.data_queue.put(data)

        self.recorder.listen_in_background(self.source, record_callback, phrase_time_limit=self.record_timeout)

        self.filepath_xlsx = "../xlsx/"
        self.filename_xlsx = "able_magok_1205.xlsx"   # 엑셀 파일 불러옴

        self.load_xlsx = load_workbook(self.filepath_xlsx+self.filename_xlsx, data_only=True)
        self.load_ws = self.load_xlsx['Sheet1']
        self.all_values = []

        for row in self.load_ws.rows:
            row_value = []
            for cell in row:
                row_value.append(cell.value)
            self.all_values.append(row_value)  # 코드 실행 시 엑셀의 모든 시나리오를 한번만 저장

        self.stt_number_pub = rospy.Publisher("/heroehs/script_number_check", Int32, queue_size=1)

        self.stt_ing_sub = rospy.Subscriber("/heroehs/able/stt_on_off", Bool, self.ActionCallback)
        self.stt_obstacle_sub = rospy.Subscriber("/heroehs/able/obstacle", Obstacle, self.ObstacleCallback)
        self.stt_number_sub = rospy.Subscriber("/heroehs/script_number", Int32, self.ScriptCallback)
        self.stt_check_sub = rospy.Subscriber("/heroehs/action/check_done", Bool, self.CheckCallback)
        self.speaker_status_sub = rospy.Subscriber("/speaker_status", Bool, self.SpeakerStatusCallback)

    def CheckCallback(self, msg):
        self.check = True

    def ScriptCallback(self, msg):
        self.check = False

    def ObstacleCallback(self, msg):
        self.obstacle = msg
        self.Decision()

    def ActionCallback(self, msg):
        self.action = msg.data

    def SpeakerStatusCallback(self, msg):
        self.speaker_active = msg.data
    
    def Decision(self): 
        if self.action and not self.speaker_active:  # STT 모드 ON 이고, 스피커가 사용 중이지 않을 때 
            if self.obstacle.mode == "human" or self.obstacle.mode == "many_people":  # 에이블 앞에 사람이 있다면
                if self.check:  #  에이블이 모션을 취하지 않고 있다면
                    self.Streaming()  #  음성인식모드 시작

    def Streaming(self):
        now = datetime.utcnow()
        if not self.data_queue.empty():
            phrase_complete = False
            if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                self.last_sample = bytes()
                phrase_complete = True
            self.phrase_time = now

            while not self.data_queue.empty():
                if self.speaker_active:  # If speaker becomes active, break the loop
                    break
                data = self.data_queue.get()
                self.last_sample += data

            if not self.speaker_active:  # Only process the audio if speaker is not active
                print("음성인식이 진행 중입니다...")  # 음성인식이 시작되었음을 알립니다.
                audio_data = sr.AudioData(self.last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                with open(self.temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Start the timer
                inference_start_time = time.time()

                result = self.audio_model.transcribe(self.temp_file, language="ko", fp16=torch.cuda.is_available())
                text = result['text'].strip()  # 모델 인퍼런스

                inference_elapsed_time = time.time() - inference_start_time

                if phrase_complete:
                    self.transcription.append(f"{text}")
                else:
                    if len(self.transcription) > 0:
                        self.transcription[-1] = f"{text}"
                    else:
                        self.transcription.append(f"{text}")

                os.system('cls' if os.name=='nt' else 'clear')

                ignore_phrases = {"구독과 좋아요", "감사합니다.", "MBC 기자", "시청해주셔서"}  # 무시할 문장들을 정의

                check_input = 0
                for line in self.transcription:
                    if not any(char.isalpha() or char.isdigit() for char in line):
                        self.transcription = []  # Initialize self.transcription
                        continue  # Skip this iteration
                    if line in ignore_phrases:  # 결과가 무시할 문장들 중 하나와 일치한다면
                        self.transcription = []  # self.transcription을 초기화하고
                        print("bad data")
                        continue  # 이번 반복을 건너뛰고 다음 반복을 진행
                    for i in range(0,(len(list(self.load_ws.rows)))):  # 매칭 시작
                        if str(self.all_values[i][1]) in line:
                            if self.all_values[i][0] == 69 :
                                self.stt_msg = random.randint(69,71)
                            else :
                                self.stt_msg = self.all_values[i][0]
                                
                            print(self.stt_msg)
                            self.stt_number_pub.publish(self.stt_msg)
                            check_input = 1
                            break
                if check_input == 1:
                    self.transcription = []
                elif check_input == 0:
                    if self.random == 0: 
                        self.stt_msg = 1000
                        self.random = 1
                    else:
                        self.stt_msg = 1001
                        self.random = 0
                    self.stt_number_pub.publish(self.stt_msg)
                    self.transcription = [] # 음성 인식이 실패했을 때에도 self.transcription을 초기화합니다.
                print(line)

                    
                    # sleep(0.25)

def main():
    rospy.init_node('able_stt_node')
    ablestt_ = AbleStt()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.on_shutdown(ablestt_.clean_up())
        print("Shutting down")

if __name__ == '__main__':
    main()


