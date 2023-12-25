import cv2
import torch
import numpy as np
import random

userScore = 0
cpuScore = 0
userMoves = []
d = np.load("rps_data.npy")
l = np.load("rps_labels.npy")
data = np.copy(d)
labels = np.copy(l)

def detectHand(frame):
    image = frame[215:515, 300:600].astype(np.float32) / 255
    adjustedImage = torch.FloatTensor(np.expand_dims(image.transpose((2,0,1)), axis=0)).to("mps")
    handDetection = classifierNet(adjustedImage)

    if len(userMoves) >= 6:
        globals()['labels'] = np.append(labels, handDetection.cpu().argmax(1).numpy())
    if handDetection.argmax(1) == 0:
        return 0, "rock"
    elif handDetection.argmax(1) == 1:
        return 1, "paper"
    elif handDetection.argmax(1) == 2:
        return 2, "scissors"
    return "Not Detected"

def cpuMove():
    cpuMove = random.randint(0,2)
    if len(userMoves) >= 6:
        previousMoves = torch.reshape(torch.FloatTensor([userMoves[-6:]]), (1,6,1))
        currentMoves = previousMoves.to("cpu")
        output = game_ai(currentMoves)
        cpuPrediction = output.argmax(1)
        if cpuPrediction == 0:
            cpuMove = 1
        elif cpuPrediction == 1:
            cpuMove = 2
        elif cpuPrediction == 2:
            cpuMove = 0
        globals()['data'] = np.vstack((data, torch.FloatTensor(userMoves[-6:]).cpu().numpy()))
        
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
    userGain = 0
    computerGain = 0
    if winner == 0:
        userGain = 1
        return userGain, computerGain, "User Wins!"
    elif winner == 1:
        computerGain = 1
        return userGain, computerGain, "Computer Wins!"
    
    return userGain, computerGain, "Tie"

videoCapture = cv2.VideoCapture(0)

classifierNet: torch.nn.Module = torch.jit.load("hand_classifier.pt")
classifierNet.eval()
game_ai: torch.nn.Module = torch.jit.load("rps_game_ai.pt")
game_ai.eval()
currentDetection = ""
currentCPUMove = ""
currentWinner = ""
while True:
    ret, frame = videoCapture.read()
    cv2.rectangle(frame, (290, 205), (610,525), (0,0,0), 5)
    cv2.putText(frame, currentDetection, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
    cv2.putText(frame, currentCPUMove, (500,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
    cv2.putText(frame, currentWinner, (1000,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
    cv2.putText(frame, "User Score: " + str(userScore) + ", " + "AI Score: " + str(cpuScore), (1400,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

    c = cv2.waitKey(1)

    if c == ord("q"):
        np.save("rps_data.npy", data)
        np.save("rps_labels.npy", labels)
        break
    elif c == ord(" "):
        computerMove = cpuMove()
        userNumber, userMove = detectHand(frame)
        currentDetection = "Your Move: " + userMove
        currentCPUMove = "Computer Move: " + computerMove
        userGain, computerGain, result = determineWinner(userMove, computerMove)
        currentWinner = "Result: " + result
        userScore += userGain
        cpuScore += computerGain
        userMoves.append(userNumber)

    cv2.imshow('Rock Paper Scissors', frame)
