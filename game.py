import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} device")

def detectHand(frame):
    image = frame[215:515, 300:600].astype(np.float32) / 255
    adjustedImage = torch.FloatTensor(np.expand_dims(image.transpose((2,0,1)), axis=0)).to(device)
    handDetection = classifierNet(adjustedImage)
    if handDetection.argmax(1) == 0:
        return "rock"
    elif handDetection.argmax(1) == 1:
        return "paper"
    elif handDetection.argmax(1) == 2:
        return "scissors"
    return "Not Detected"

def cpuMove():
    cpuMove = random.randint(0,2)
    cpuHand = ""
    if cpuMove == 0:
        cpuHand = "rock"
    elif cpuMove == 1:
        cpuHand = "paper"
    elif cpuMove == 2:
        cpuHand = "scissors"
    return cpuHand

def determineWinner(userMove, cpuMove):
    moves = [userMove, cpuMove]
    winner = -1
    if "rock" in moves and "paper" in moves:
        winner = moves.index("paper")
    elif "rock" in moves and "scissors" in moves:
        winner = moves.index("rock")
    elif "paper" in moves and "scissors" in moves:
        winner = moves.index("scissors")
    
    if winner == 0:
        return "User Wins!"
    elif winner == 1:
        return "Computer Wins!"
    return "Tie"

videoCapture = cv2.VideoCapture(0)

userScore = 0
cpuScore = 0

classifierNet: torch.nn.Module = torch.jit.load("hand_classifier.pt")
classifierNet.eval()
currentDetection = ""
currentCPUMove = ""
currentWinner = ""
while True:
    ret, frame = videoCapture.read()
    cv2.rectangle(frame, (290, 205), (610,525), (0,0,0), 5)
    cv2.putText(frame, currentDetection, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
    cv2.putText(frame, currentCPUMove, (600,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
    cv2.putText(frame, currentWinner, (1100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

    c = cv2.waitKey(1)

    if c == ord("q"):
        break
    elif c == ord(" "):
        userMove = detectHand(frame)
        computerMove = cpuMove()
        currentDetection = "Your Move: " + userMove
        currentCPUMove = "Computer Move: " + computerMove
        currentWinner = "Result: " + determineWinner(userMove, computerMove)
    cv2.imshow('Rock Paper Scissors', frame)