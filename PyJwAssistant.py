import pyaudio
import math
import struct
import wave
import time
import datetime
import os
import sys
import select
from faster_whisper import WhisperModel
from llama_cpp import Llama
import asyncio
import queue
import threading
import io

#TODO:
# *** done *** add conversation context.
# *** done *** speed up NLP. (faster whisper?)
# *** done *** TTS on a per sentence basis.
# *** done *** Set up sleep mode
# *** done *** change from saving to file to using an object.
# *** done *** go to sleep after X seconds
# *** done *** add option to read/save transcript.
# *** done *** basic scratchPad list tool
# *** done *** add option to interrupt output.
#- add RAG integration
#- buffer a rolling window for recording.
#- set up self wake/reminders
#- change sleep behavior to require keyboard input to wake.

### configuration.start
TIMEOUT_LENGTH = 2
SLEEP_TIMEOUT_SECONDS = 15

contextWindowSize=8192

MODEL_PATH="/Users/junsunwhang/Development/Models/Meta-Llama-3-8B-Instruct.Q8_0.gguf"
#MODEL_PATH="/Users/junsunwhang/Development/Models/Llama-3-8B-Instruct-32k-v0.1.fp16.gguf"
#MODEL_PATH="/Users/junsunwhang/Development/Models/Meta-Llama-3-8B-Instruct.fp16.gguf"

workingDirectory = '/Users/junsunwhang/Test'
historyPath = os.path.join(workingDirectory, "history")
scratchPadListPath = os.path.join(workingDirectory, "scratchPad")

storedConversationMessages=50 #each iteration is 2, 1 for the user, 1 for the LLM

AGENT_NAME = "assistant"
#Phrases to activation actions.
PHRASE_SLEEP = "ok, going to sleep"
PHRASE_SAVE_HISTORY = "ok, saving conversation history"
PHRASE_LOAD_HISTORY = "ok, loading conversation history for use in your next prompt"
PHRASE_PERSIST_SCRATCHPAD = "ok, persisting in-memory scratchpad"

### configuration.end
Threshold = 10

SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

swidth = 2

MODE_AWAKE = "awake"
MODE_SLEEPING = "sleeping"

sentenceQueue = queue.Queue()
currentlySpeaking = False
interrupt_requested = False

class Recorder:
    @staticmethod
    def rms(frame):
        count = len(frame) / swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self):
        print("\033[36m=== PyAssistant Chat ===\033[0m")
        print("initializing...")
        print(f"Using model: {MODEL_PATH}")
        print("Press 'Enter' to interrupt a response, and 'Q then enter' to quit the program.")
        print(f"You can ask the LLM to go to sleep, or wake when addressing the LLM as '{AGENT_NAME}'.\n"\
              f"The LLM will sleep automatically while in listen mode after {SLEEP_TIMEOUT_SECONDS} seconds.")

        self.historyToAdd = ""
        self.attentionMode=MODE_AWAKE #sleeping | listening
        self.sleepNotificationTripped=False

        self.conversation_history = []
        self.model = Llama(model_path=MODEL_PATH,
                      n_ctx=contextWindowSize,  # context window size
                      n_gpu_layers=-1,  # enable GPU
                      use_mlock=True,  # enable memory lock so not swap
                      use_mmap=False,
                      verbose=False)
        self.whisperModel = WhisperModel("base.en", device="cpu", compute_type="int8")

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=chunk)

        print("complete.")

    async def record(self):
        print('\033[92mRecording...\033[0m')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:
            data = self.stream.read(chunk)
            if self.rms(data) >= Threshold: end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)

        # Add a character to console notify recording is done.
        print("\033[93m>>>")
        os.system(f"afplay /System/Library/Sounds/Submarine.aiff")

        # Create an in-memory buffer
        audio_data = b''.join(rec)
        buffer = io.BytesIO(audio_data)

        # Create a temporary WAV file in memory
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(audio_data)
            wav_buffer.seek(0)

            start_time = datetime.datetime.now()
            textContents = ""
            segments, info = self.whisperModel.transcribe(wav_buffer, beam_size=5)
            for segment in segments:
                textContents += segment.text

        textContents = textContents.strip()
        print(f"\033[93m{textContents}\033[0m")

        userQuery = textContents.strip()

        #reset stream
        resetTask=asyncio.create_task(self.resetStream())

        if AGENT_NAME in userQuery.lower():
            self.attentionMode = MODE_AWAKE
            self.sleepNotificationTripped = False
        if self.attentionMode == MODE_AWAKE:
            self.queryLlm(userQuery)

    async def resetStream(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=chunk)

    def queryLlm(self, userQuery):
        global currentlySpeaking
        start_time = datetime.datetime.now()

        systemPrompt = "If a question does not make any sense, explain why instead of answering something not correct.\n" \
                       "If you don't know the answer to a question, simply state that you do not know the answer.\n" \
                       f"If the user tells you to sleep, respond with '{PHRASE_SLEEP}'\n" \
                       f"If the user asks you to save the conversation history, respond with '{PHRASE_SAVE_HISTORY}'\n" \
                       f"If the user asks you to load the conversation history, respond with '{PHRASE_LOAD_HISTORY}'\n" \
                       f"If the user asks you to persist the scratch pad, start your response " \
                       f"with '{PHRASE_PERSIST_SCRATCHPAD}:' followed by the information from this current prompt."

        if (len(self.historyToAdd)>0):
            systemPrompt += "Add the following context from a previous session: " + self.historyToAdd
            self.historyToAdd=""

        messages = [
                {"role": "system", "content": f"{systemPrompt}"},
            ]

        # Add the last 5 interactions to the messages
        messages.extend(self.conversation_history[-storedConversationMessages:])

        # Add the current user query
        messages.append({"role": "user", "content": f"{userQuery}"})

        output = self.model.create_chat_completion(
            messages=messages,
            stream=True
        )

        lineString = ""
        full_response = ""
        for chunk in output:
            if interrupt_requested:
                break
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                contentString = delta['content']
                #print(f"\033[94m{contentString}\033[0m",end='') #debug
                print(f"{contentString}", end='')  # debug
                lineString += contentString
                full_response += contentString
                if contentString in ['.', '!', '?', '\n']:
                    sentenceQueue.put(lineString.strip())
                    lineString = ""

        if len(lineString)>0:
            sentenceQueue.put(lineString.strip())
            lineString = ""

        print("\n")

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": userQuery})
        self.conversation_history.append({"role": "assistant", "content": full_response.strip()})

        # change modes if needed.
        if (PHRASE_SLEEP in full_response.lower()):
            self.attentionMode=MODE_SLEEPING

        # save and load conversation history.
        if (PHRASE_SAVE_HISTORY in full_response.lower()):
            #collect conversation history.
            historyText = ""
            for message in self.conversation_history:
                role = message["role"]
                content = message["content"]
                if role == "user":
                    historyText += f"\nUser:{content}"
                elif role == "assistant":
                    historyText += f"\nAssistant:{content}"
            #save the contents to a file.
            self.write_to_file(historyPath + ".txt", historyText)
            #save another version with time stamp.
            current_datetime = datetime.datetime.now()
            timestamp = current_datetime.strftime("%Y_%m_%d__%H_%M_%S")
            self.write_to_file(historyPath + "_" + timestamp + ".txt", historyText)

        if (PHRASE_LOAD_HISTORY in full_response.lower()):
            #load string from history file.
            self.historyToAdd = self.read_from_file(historyPath + ".txt")

        if (PHRASE_PERSIST_SCRATCHPAD in full_response.lower()):
            itemList = full_response[len(PHRASE_PERSIST_SCRATCHPAD):]
            self.write_to_file(scratchPadListPath, itemList)

        # Keep only the last 5 interactions (10 messages) by default.
        self.conversation_history = self.conversation_history[-storedConversationMessages:]

        #wait until all the text has been spoken.
        while (not sentenceQueue.empty()) or (currentlySpeaking==True):
            if interrupt_requested and currentlySpeaking==False:
                break
            time.sleep(.2)

    def write_to_file(self, filename, content):
        try:
            with open(filename, 'w') as file:
                file.write(content)
                file.close()
            print(f"Content successfully written to {filename}")
        except IOError as e:
            print(f"An error occurred while writing to the file: {e}")

    def read_from_file(self, filename):
        try:
            with open(filename, 'r') as file:
                content = file.read()
                file.close()
            return content
        except IOError as e:
            print(f"An error occurred while reading the file: {e}")
            return None

    def listen(self):
        global interrupt_requested

        print('\n\033[94m___Listening...\033[0m',end='')
        os.system(f"afplay /System/Library/Sounds/Blow.aiff")
        last_action_time = time.time()

        while True:
            interrupt_requested = False  # Reset the interrupt.
            current_time = time.time() # automatically go to sleep after X seconds.
            if current_time - last_action_time > SLEEP_TIMEOUT_SECONDS:
                self.attentionMode = MODE_SLEEPING
                if (self.sleepNotificationTripped==False):
                    print("\033[95msleeping...\033[0m")
                    self.sleepNotificationTripped=True

            input = self.stream.read(chunk, exception_on_overflow=False)
            rms_val = self.rms(input)
            if rms_val > Threshold:
                asyncio.run(self.record())
                print('\n\033[94m___Listening...\033[0m',end='')
                os.system(f"afplay /System/Library/Sounds/Blow.aiff")
                last_action_time = time.time()

class AsyncTTS:
    _instance = None
    _lock = threading.Lock()

    # Function to handle printing as a singleton
    def TTS_handler(self):
        global interrupt_requested
        while True:
            sentence = sentenceQueue.get()
            if interrupt_requested:
                sentenceQueue.queue.clear()
                continue
            self.speakSentence(sentence)
            #print(f"TTS_handler: {sentence}")
            time.sleep(0.1)

    def __init__(self):
        self.tts_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def speakSentence(self, sentence):
        global currentlySpeaking
        currentlySpeaking=True
        escapedSentence = sentence.replace('"','\\"')
        os.system(f'say "{escapedSentence}"')
        #print("starting speech")
        # command = ['say', sentence]
        # subprocess.run(command, check=True)
        #print("ending speech")
        currentlySpeaking=False

def check_for_interrupt():
    global interrupt_requested
    if select.select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.read(1)
        if key.lower() == 'q':
            print("\033[31mQuitting program.\033[0m")
            return True
        elif key == '\n':
            print("\033[31mInterrupting...\033[0m")

            interrupt_requested = True
    return False


llm = Recorder()
tts = AsyncTTS()
llmThread = threading.Thread(target=llm.listen, daemon=True)
ttsThread = threading.Thread(target=tts.TTS_handler, daemon=True)

llmThread.start()
ttsThread.start()

while True:
    if check_for_interrupt():
        break
