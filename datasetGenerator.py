import numpy as np
# import pandas as pd

np.set_printoptions(suppress = True)

deltaT = 1

time = np.arange(0, 101, deltaT)
initialVelocity = 100
initialPosition = 0
acceleration = 2

idealVelocity = initialVelocity + (2 * time)

#print(idealVelocity)

# Based off the kinematics equation: x_f = x_i + (v_i * deltaT) + (0.5 * a * (deltaT)^2)

idealValues_Matrix = np.empty([2, 100])

# Sets the very first position and velocity to the given initial values
idealValues_Matrix[0, 0] = initialPosition
idealValues_Matrix[1, 0] = initialVelocity

# print(idealValues_Matrix.shape[1])


def generatePositions(initialPosition, initialVelocity, acceleration, iter):

    # Based off the kinematics equation: x_f = x_i + (v_i * deltaT) + (0.5 * a * (deltaT)^2)
    finalPosition = initialPosition + (initialVelocity * deltaT) + (0.5 * acceleration * (deltaT**2))

    idealValues_Matrix[0, iter] = finalPosition

    # Base case: if the iteration is equal to the length of our values Matrix we don't want the recursion to continue
    if(iter == idealValues_Matrix.shape[1]-1):

        return

    # make sure we are counting the iteration number
    iter += 1

    # RECURSION!!!!!!
    return generatePositions(finalPosition, idealVelocity[iter], acceleration, iter)

def generateVelocities(initialVelocity, acceleration, iter):\

    # Based off the kinematics equation: v_f = v_i + (a * deltaT)
    finalVelocity = initialVelocity + (acceleration * deltaT)

    idealValues_Matrix[1, iter] = finalVelocity

    # Base case: if the iteration is equal to the length of our values Matrix we don't want the recursion to continue
    if(iter == idealValues_Matrix.shape[1]-1):

        return
    
    # make sure we are counting the iteration number
    iter += 1

    # RECURSION!!!!!!
    return generateVelocities(finalVelocity, acceleration, iter)

generatePositions(initialPosition, initialVelocity, acceleration, 1)
generateVelocities(initialVelocity, acceleration, 1)

#print(idealValues_Matrix)
#print(idealValues_Matrix.shape[1])

positionNoise = np.random.normal(-5, 5, size=(idealValues_Matrix.shape[1]))

velocityNoise = np.random.normal(-1, 1, size=(idealValues_Matrix.shape[1]))

#print(noise)
measuredPositions = idealValues_Matrix[0] + positionNoise

measuredVelocities = idealValues_Matrix[1] + velocityNoise

# Ensures that the first measured position and velocity is always 0 and 100 respectively
measuredPositions[0] = initialPosition
measuredVelocities[0] = initialVelocity

observedValues = np.empty([idealValues_Matrix.shape[1], 2]);

for i in range (0, idealValues_Matrix.shape[1]):
    observedValues[i, 0] = measuredPositions[i]
    observedValues[i, 1] = measuredVelocities[i]


# print(measuredPositions)
# print(measuredVelocities)



