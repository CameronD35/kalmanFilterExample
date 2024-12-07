import numpy as np
import datasetGenerator as DG

deltaT = 1
acceleration = 2
obsPosErr = 10
obsVelErr = 2
proPosErr = 4
proVelErr = 0.5

obsErr = np.array([[obsPosErr], [obsPosErr], [obsPosErr], [obsVelErr], [obsVelErr], [obsVelErr]])
proErr = np.array([[proPosErr], [proPosErr], [proPosErr], [proVelErr], [proVelErr], [proVelErr]])
#print(obsErr)

def createKalmanFilter(deltaT, acceleration, dimensions, dataset, observationErrArray, processErrArray):

    # Sets the A matrix based on the dimensions we are working in (dimensions variable)
    A = np.zeros([dimensions, dimensions])

    for i in range(0, dimensions):
        A[i, i] = 1

        if(i < A.shape[0] - 3):
            A[i, i+3] = deltaT

    # The width of B is half the dimensions since the control variable (in kinematiacs it is acceleration) only affects the axes present (x, y, z)
    B_width = int(dimensions/2)
    B = np.zeros([dimensions, B_width])


    for i in range(0, dimensions):

        if (i <= (B_width) - 1):
            B[i, i] = (0.5 * (deltaT**2))
        else:
            B[i, i - (B_width)] = deltaT


    # The H matrix is an identity matrix in this case of n dimensions where n equal to the dimensions variable
    H = np.zeros([dimensions, dimensions])

    for i in range(0, dimensions):

        H[i, i] = 1

    # takes the first measurement in the dataset and converts it to a n by 1 matrix
    initialMeasurement = np.zeros([dimensions, 1])

    for i in range(0, dimensions):
        initialMeasurement[i, 0] = dataset[0][i]

    # print(initialMeasurement)

    # takes the second measurement in the dataset and converts it to a n by 1 matrix
    secondMeasurement = np.zeros([dimensions, 1])

    for i in range(0, dimensions):
        secondMeasurement[i, 0] = dataset[1][i]


    

    generalizedKalman(deltaT, acceleration, initialMeasurement, secondMeasurement, processErrArray, observationErrArray, dataset, -1, A, B, H, 1)

#print(observedValues)

def generalizedKalman(deltaT, acceleration, initialMeasurementArray, secondMeasurementArray, processErrArray, observationErrArray, dataset, prevProcessCVMatrix, A, B, H, iter):

    #print(dataset)
    #print(len(dataset))

    # A
    A_Matrix = A

    # B
    B_Matrix = B

    # H
    H_Matrix = H

    # X_kp
    predictedStateMatrix = getPredStateMatrix(deltaT, acceleration, initialMeasurementArray, A_Matrix, B_Matrix, -1)

    # P_k-1
    processCovarianceMatrix = 0

    # If we already have a previous process covariance matrix, we should not create an initial one
    if (type(prevProcessCVMatrix) is int):
        processCovarianceMatrix = getInitProcessCVMatrix(processErrArray)
    else:
        #print(prevProcessCVMatrix)
        processCovarianceMatrix = prevProcessCVMatrix

    # P_kp
    predictedProcessCovarianceMatrix = getPredProcessCVMatrix(processCovarianceMatrix, A_Matrix, -1)

    # K
    kalminGain = getKalmanGain(predictedProcessCovarianceMatrix, observationErrArray, H_Matrix)

    # Y_k
    newObservation = getNewObservation(secondMeasurementArray, -1, H)
    # X_k
    filteredState = calculateFilteredState(kalminGain, predictedStateMatrix, newObservation, H_Matrix)

    # P_k
    updatedProcessCovarianceMatrix = updateProcessCVMatrix(kalminGain, H_Matrix, predictedProcessCovarianceMatrix)

    #print(updatedProcessCovarianceMatrix)

    print(f'Iteration #: {iter}\nThe predicted x-position is: {filteredState[0, 0]}.\nThe predicted x-velocity is: {filteredState[3, 0]}\nThe new process covariance matrix is:\n{updatedProcessCovarianceMatrix}\n\n')

    # Removes the very first item from the dataset tuple to set up for recursion
    slicedDataset = dataset[1:len(dataset)]
    

    # Base case: Since the dataset would not have an index of 1 (see that index 0 and 1 are necessary for the filter), the function is exited 
    if (len(slicedDataset) <= 1):
        print('Kalmin filter finished')
        return
    
    # The below code sets up the new measurements by setting the previous current to previous and the next point to current
    previousMeasurement = np.zeros([filteredState.shape[0], 1])


    for i in range(0, A_Matrix.shape[0]):
        previousMeasurement[i, 0] = slicedDataset[0][i]


    currentMeasurement = np.zeros([A_Matrix.shape[0], 1])

    for i in range(0, A_Matrix.shape[0]):
        currentMeasurement[i, 0] = slicedDataset[1][i]
    
    iter += 1

    # RECURSION!
    return generalizedKalman(deltaT, acceleration, previousMeasurement, currentMeasurement, processErrArray, observationErrArray, slicedDataset, updatedProcessCovarianceMatrix, A, B, H, iter)


# Predicted State Matrix
def getPredStateMatrix(deltaT, acceleration, initialMeasurementArray, A, B, error_Matrix):
    
    # A
    A_Matrix = A

    #print(A_Matrix)

    # Its adding indecies 0 and 2 for some reason?

    # X_k-1
    previousState_Matrix = initialMeasurementArray
    print('i', previousState_Matrix)
    # B
    B_Matrix = B

    # mu_k
    controlVariable_Matrix = np.zeros([B_Matrix.shape[1], 1])

    for i in range(0, B_Matrix.shape[1]):
        controlVariable_Matrix[i, 0] = acceleration
        
    # A * X_k-1
    formattedState_Matrix = np.dot(A_Matrix, previousState_Matrix)
    print('a', A_Matrix)
    print('s', formattedState_Matrix)

    # B * mu_k
    formattedControlVariable_Matrix = np.dot(B_Matrix, controlVariable_Matrix)
    print('b', formattedControlVariable_Matrix)
    # Adds the above two matricies
    predictedStateMatrix = np.add(formattedState_Matrix, formattedControlVariable_Matrix)
    print('p:',predictedStateMatrix)
    # w_k
    if (error_Matrix != -1):
        return np.add(predictedStateMatrix, error_Matrix)

    return predictedStateMatrix

# Initial Process Covariance Matrix
def getInitProcessCVMatrix(processErrorArray):
    #print('getting process covariance matrix')
    
    processErrorArray_Size = processErrorArray.shape

    # P_k-1
    processCovariance_Matrix = np.zeros([processErrorArray_Size[0], processErrorArray_Size[0]])

    """
    [
    [x^2, xy],
    [yx, y^2]
    ]

    [
    [x^2, xy, xz],
    [yx, y^2, yz],
    [zx, zy, z^2]
    ]
    
    """

    # Corrects the off-diagonal number. This indicates that our error in velocity does not effect the error in position and vice versa. (Done for simplicity, may not reflect reality)
    # print(processErrorArray)
    # print(processCovariance_Matrix)
    for i in range(0, processErrorArray_Size[0]):

        for j in range (0, processErrorArray_Size[0]):
            # print (i, '', j)
            processCovariance_Matrix[i, j] = processErrorArray.item(i) * processErrorArray.item(j)

            if (i != j):
                processCovariance_Matrix[i, j] = 0
            # if (i == j):
            #     processCovariance_Matrix[i, j] = processErrorArray[0, i]
            # else:
            #   processCovariance_Matrix[i, j] = processErrorArray[0, i] * processErrorArray[0, j]
            
    # processCovariance_Matrix = np.matrix([
    #     [processPositionError**2, processPositionError*processVelocityError],
    #     [processPositionError*processVelocityError, processVelocityError**2]
    # ])

    # processCovariance_Matrix[0, 1] = 0
    # processCovariance_Matrix[1, 0] = 0

    #print(processCovariance_Matrix)

    return processCovariance_Matrix

# Predicted Process Covariance Matrix
def getPredProcessCVMatrix(initProcessCVMatrix, A, Q):
    
    #A
    A_Matrix = A

    # A * P_k-1 * transpose(A)
    #print(initProcessCVMatrix)
    #print(A_Matrix)
    firstTerm = np.dot(np.dot(A_Matrix, initProcessCVMatrix), np.transpose(A_Matrix))

    # Q
    qError = Q

    # Since we set the off-diagonals to 0 in the previous step, we do it again to remain consistent
    for i in range(0, firstTerm.shape[0]):

        for j in range (0, firstTerm.shape[0]):
            # print (i, '', j)

            if (i != j):
                firstTerm[i, j] = 0

    # If we assume no error in our prediction process, we don't add to the firstTerm matrix
    if (qError != -1):

        finalMatrix = firstTerm + Q

        return finalMatrix
    
    else:

        return firstTerm

# Calculates the kalman gain
def getKalmanGain(predProcessCVMatrix, observationErrorArray, H):
    
    # H
    H_Matrix = H

    # transpose(H)
    H_Transposed = np.transpose(H_Matrix)

    # P_kp * transpose(H)
    numerator = np.dot(predProcessCVMatrix, H_Transposed)

    # H * P_kp * H^T
    denominatorFirstTerm = np.dot(np.dot(H_Matrix, predProcessCVMatrix), H_Transposed)




    # R, here we introduce the errors (either estimated or given by datasheets, us, etc.)

    observationErrorArray_Size = observationErrorArray.shape

    R_Matrix = np.zeros([observationErrorArray_Size[0], observationErrorArray_Size[0]])

    # Off-diagonals are set to 1
    for i in range(0, observationErrorArray_Size[0]):

        for j in range (0, observationErrorArray_Size[0]):

            R_Matrix[i, j] = observationErrorArray.item(i) * observationErrorArray.item(j)

            if (i != j):
                R_Matrix[i, j] = 1
    # R_Matrix = [
    #     [observationPositionError**2, 1],
    #     [1, observationVelocityError**2]
    # ]

    # denominatorFirstTerm + R
    denominator = np.add(denominatorFirstTerm, R_Matrix)

    # numerator/denominator
    kalminGain = np.divide(numerator, denominator)

    return kalminGain

# gets the next observation
def getNewObservation(secondObservationArray, error_Matrix, H):
    
    # C
    H_Matrix = H

    # Y_km
    measuredValues_Matrix = secondObservationArray

    # H * Y_km
    reformattedY_Matrix = np.dot(H_Matrix, measuredValues_Matrix)

    # Z, like other errors we may or may not need this value. If we do, we add it to the above matrix, otherwise we ignore it
    if(error_Matrix != -1):
        return np.add(reformattedY_Matrix + error_Matrix)
    else:
        return reformattedY_Matrix

# Calculates the kalmanized state to filter out noise
def calculateFilteredState(kalminGain, predState_Matrix, observation_Matrix, H):
    # H * X_kp
    modifiedPredState_Matrix = np.dot(H, predState_Matrix)
    # Y_k - (H * X_kp)
    predObsDifference_Matrix = np.subtract(observation_Matrix, modifiedPredState_Matrix)
    # Multiply above matrix by kalmin gain (K * Above Matrix)
    kalminized = np.dot(kalminGain, predObsDifference_Matrix)
    # Added the 'kalminized' (Idk if that's a word) matrix to the original predicted state matrix
    finalCalculatedState = np.add(kalminized, predState_Matrix)

    return(finalCalculatedState)

# updates the covariance matrix for the next iteration
def updateProcessCVMatrix(kalminGain, H_Matrix, predProcessCV_Matrix):

    kalminGainHeight = kalminGain.shape[0]
    
    # I
    identity_Matrix = np.zeros((kalminGainHeight, kalminGainHeight))

    for i in range(0, kalminGainHeight):
        identity_Matrix[i, i] = 1


    # K * H
    reformattedKalminGain_Matrix = np.dot(kalminGain, H_Matrix)

    # I - above matrix
    processCVFactor_Matrix = np.subtract(identity_Matrix, reformattedKalminGain_Matrix)

    # Above Matrix * P_kp
    updatedProcessCV_Matrix = np.dot(processCVFactor_Matrix, predProcessCV_Matrix)

    return updatedProcessCV_Matrix

# (Position, Velocity)
dataset = DG.values_Matrix

createKalmanFilter(1, 2, 6, dataset, obsErr, proErr)
