import random
import numpy

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

currentMoves = []

allMoves = []

while len(allMoves) < 500:
    userMove = input("Enter your move: ")
    userWord = "rock"
    userNumber = 0
    if userMove == "r":
        userWord = "rock"
        userNumber = 0
    elif userMove == "p":
        userWord = "paper"
        userNumber = 1
    elif userMove == "s":
        userWord = "scissors"
        userNumber = 2
    computerMove = cpuMove()
    print("Computer Selected:", computerMove)
    print(determineWinner(userWord, computerMove))
    if len(currentMoves) == 3:
        currentMoves.pop(0)
        currentMoves.append(userNumber)
        allMoves.append(currentMoves.copy())
        print("Recorded Data Number:", len(allMoves))
    else:
        currentMoves.append(userNumber)

data = numpy.array(allMoves)
numpy.save("rps_data.npy", data)