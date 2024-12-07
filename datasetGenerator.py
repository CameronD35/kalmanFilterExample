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
def createEmptyMatrix(pointCount, dimensions):
    emptyMatrix = np.empty([pointCount, dimensions])

    for i in range(0, dimensions):
        'hi'
    return emptyMatrix

values_Matrix = createEmptyMatrix(5, 6)

# Sets the very first position and velocity to the given initial values
values_Matrix[0, 0] = initialPosition

values_Matrix[0, 1] = initialPosition

values_Matrix[0, 2] = initialPosition

values_Matrix[0, 3] = initialVelocity

values_Matrix[0, 4] = initialVelocity

values_Matrix[0, 5] = initialVelocity

# [xPos, yPos, zPos, xVel, yVel, zVel]

#print(values_Matrix)


# print(idealValues_Matrix.shape[1])


"""
Axis #:
x = 0
y = 1
z = 2
"""

def generatePositions(initialPosition, initialVelocity, acceleration, iter, axisNumber):

    # Base case: if the iteration is equal to the length of our values Matrix we don't want the recursion to continue
    if(iter == values_Matrix.shape[1]-1):

        return

    # Based off the kinematics equation: x_f = x_i + (v_i * deltaT) + (0.5 * a * (deltaT)^2)
    positionNoise = np.random.normal(-5, 5)
    #print(positionNoise)
    finalPosition = initialPosition + (initialVelocity * deltaT) + (0.5 * acceleration * (deltaT**2)) + positionNoise

    #print(finalPosition)

    values_Matrix[iter, axisNumber] = finalPosition


    # make sure we are counting the iteration number
    iter += 1

    # RECURSION!!!!!!
    return generatePositions(finalPosition, idealVelocity[iter], acceleration, iter, axisNumber)

def generateVelocities(initialVelocity, acceleration, iter, axisNumber):

    # Base case: if the iteration is equal to the length of our values Matrix we don't want the recursion to continue
    if(iter == values_Matrix.shape[1]-1):

        return


    # Based off the kinematics equation: v_f = v_i + (a * deltaT)
    velocityNoise = np.random.normal(-1, 1)

    finalVelocity = initialVelocity + (acceleration * deltaT) + velocityNoise


    values_Matrix[iter, axisNumber + 3] = finalVelocity
    
    
    # make sure we are counting the iteration number
    iter += 1
    print
    # RECURSION!!!!!!
    return generateVelocities(finalVelocity, acceleration, iter, axisNumber)

generatePositions(initialPosition, initialVelocity, acceleration, 1, 0)
generatePositions(initialPosition, initialVelocity, acceleration, 1, 1)
generatePositions(initialPosition, initialVelocity, acceleration, 1, 2)
generateVelocities(initialVelocity, acceleration, 1, 0)
generateVelocities(initialVelocity, acceleration, 1, 1)
generateVelocities(initialVelocity, acceleration, 1, 2)

print('test:', values_Matrix)

#print(idealValues_Matrix)
#print(idealValues_Matrix.shape[1])

#print(noise)
# measuredPositions = idealValues_Matrix[0] + positionNoise

# measuredVelocities = idealValues_Matrix[1] + velocityNoise

# Ensures that the first measured position and velocity is always 0 and 100 respectively
# values_Matrix[0] = initialPosition
# values_Matrix[0] = initialVelocity

# observedValues = np.empty(values_Matrix.shape);
# print(observedValues)

# for i in range (0, values_Matrix.shape[0]):
#     observedValues[i, 0] = values_Matrix[i, 0]
#     observedValues[i, 1] = values_Matrix[i, 1]

#print(values_Matrix)
# print(measuredPositions)
# print(measuredVelocities)



