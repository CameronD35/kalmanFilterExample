import numpy as np
import datasetGenerator as DG

deltaT = 1
acceleration = 2
obsPosErr = 5
obsVelErr = 1
proPosErr = 4
proVelErr = 0.5

obsErr = np.array([[obsPosErr], [obsVelErr]])
proErr = np.array([[proPosErr], [proVelErr]])
#print(obsErr)

def createKalmanFilter(deltaT, acceleration, dimensions, dataset, observationErrArray, processErrArray):

    A = np.empty([dimensions, dimensions])

    for i in range(0, dimensions):
        A[i, i] = 1

        if(i < A.shape[0] - 2):
            A[i, i+2] = deltaT


    B_width = int(dimensions/2)
    B = np.empty([dimensions, B_width])

    for i in range(0, dimensions):

        if (i <= (B_width) - 1):
            B[i, i] = (0.5 * (deltaT**2))
        else:
            B[i, i - (B_width)] = deltaT


    H = np.empty([dimensions, dimensions])

    for i in range(0, dimensions):

        H[i, i] = 1

    # print(A)
    # print(B)
    # print(H)

    initialMeasurement = np.empty([dimensions, 1])

    for i in range(0, dimensions):
        initialMeasurement[i, 0] = dataset[0][i]

    # print(initialMeasurement)


    secondMeasurement = np.empty([dimensions, 1])

    for i in range(0, dimensions):
        secondMeasurement[i, 0] = dataset[1][i]

    # print(secondMeasurement)


    generalizedKalman(deltaT, acceleration, initialMeasurement, secondMeasurement, processErrArray, observationErrArray, dataset, -1, A, B, H, 0)

observedValues = DG.observedValues

#print(observedValues)

def generalizedKalman(deltaT, acceleration, initialMeasurementArray, secondMeasurementArray, processErrArray, observationErrArray, dataset, prevProcessCVMatrix, A, B, H, iter):

    
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

    print(f'Iteration #: {iter}\nThe predicted position is: {filteredState[0, 0]}.\nThe predicted velocity is: {filteredState[1, 0]}\nThe new process covariance matrix is:\n{updatedProcessCovarianceMatrix}')

    # Removes the very first item from the dataset tuple to set up for recursion
    #print(dataset)
    slicedDataset = dataset[1:len(dataset)]
    #print(slicedDataset)

    # Base case: Since the dataset would not have an index of 1 (see that index 0 and 1 are necessary for the filter), the function is exited 
    if (len(slicedDataset) <= 1):
        print('Kalmin filter finished')
        return
    
    initialMeasurement = np.array([[filteredState[0, 0]], [filteredState[1, 0]]])

    for i in range(0, A_Matrix.shape[0]):
        initialMeasurement[i, 0] = slicedDataset[0][i]

    #print(initialMeasurement)
    # print(initialMeasurement)


    secondMeasurement = np.empty([A_Matrix.shape[0], 1])

    for i in range(0, A_Matrix.shape[0]):
        secondMeasurement[i, 0] = slicedDataset[1][i]
    #print(secondMeasurement)
    iter += 1

    #print(iter)
    # RECURSION!
    return generalizedKalman(deltaT, acceleration, initialMeasurement, secondMeasurement, processErrArray, observationErrArray, slicedDataset, updatedProcessCovarianceMatrix, A, B, H, iter,)

# def kinematicsKalman(deltaT, acceleration, measuredPosition, measuredVelocity, processPositionError, processVelocityError, observationPositonError,
#                      observationVelocityError, secondMeasuredPosition, secondMeasuredVelocity, dataset, prevProcessCVMatrix, A, B, H):

    
#     #print(len(dataset))

#     # A
#     A_Matrix = A

#     # B
#     B_Matrix = B

#     # H
#     H_Matrix = H

#     # X_kp
#     predictedStateMatrix = getPredStateMatrix(deltaT, acceleration, measuredPosition, measuredVelocity, A_Matrix, B_Matrix, -1)

#     # P_k-1
#     processCovarianceMatrix = 0

#     # If we already have a previous process covariance matrix, we should not create an initial one
#     if (type(prevProcessCVMatrix) is int):
#         processCovarianceMatrix = getInitProcessCVMatrix(processPositionError, processVelocityError)
#     else:
#         #print(prevProcessCVMatrix)
#         processCovarianceMatrix = prevProcessCVMatrix

#     # P_kp
#     predictedProcessCovarianceMatrix = getPredProcessCVMatrix(processCovarianceMatrix, A_Matrix, -1)

#     # K
#     kalminGain = getKalmanGain(predictedProcessCovarianceMatrix, observationPositonError, observationVelocityError, H_Matrix)

#     # Y_k
#     newObservation = getNewObservation(secondMeasuredPosition, secondMeasuredVelocity, -1)

#     # X_k
#     filteredState = calculateFilteredState(kalminGain, predictedStateMatrix, newObservation, H_Matrix)

#     # P_k
#     updatedProcessCovarianceMatrix = updateProcessCVMatrix(kalminGain, H_Matrix, predictedProcessCovarianceMatrix)

#     #print(updatedProcessCovarianceMatrix)

#     print(f'The predicted position is: {filteredState[0, 0]}.\nThe predicted velocity is: {filteredState[1, 0]}\nThe new process covariance matrix is:\n{updatedProcessCovarianceMatrix}')

#     # Removes the very first item from the dataset tuple to set up for recursion
#     slicedDataset = dataset[1:len(dataset)]
#     #print(slicedDataset)

#     # Base case: Since the dataset would not have an index of 1 (see that index 0 and 1 are necessary for the filter), the function is exited 
#     if (len(slicedDataset) <= 1):
#         print('Kalmin filter finished')
#         return
    
#     # RECURSION!
#     return kinematicsKalman(deltaT, acceleration, filteredState[0, 0], filteredState[1, 0], 20, 5, 25, 6, slicedDataset[1][0], slicedDataset[1][1], slicedDataset, updatedProcessCovarianceMatrix, A_Matrix, B_Matrix, H_Matrix)


# Predicted State Matrix
def getPredStateMatrix(deltaT, acceleration, initialMeasurementArray, A, B, error_Matrix):
    #print('getting predicted state matrix')
    
    # A
    A_Matrix = A

    #print(A_Matrix)

    # X_k-1
    previousState_Matrix = initialMeasurementArray

    # B
    B_Matrix = B

    # mu_k
    controlVariable_Matrix = np.array([
        [acceleration]
    ])
    
    # A * X_k-1
    formattedState_Matrix = np.dot(A_Matrix, previousState_Matrix)
    #print(formattedStateMatrix)

    # B * mu_k
    formattedControlVariable_Matrix = np.dot(B_Matrix, controlVariable_Matrix)

    # Adds the above two matricies
    predictedStateMatrix = np.add(formattedState_Matrix, formattedControlVariable_Matrix)

    # w_k
    if (error_Matrix != -1):
        return np.add(predictedStateMatrix, error_Matrix)

    return predictedStateMatrix

# Initial Process Covariance Matrix
def getInitProcessCVMatrix(processErrorArray):
    #print('getting process covariance matrix')
    
    processErrorArray_Size = processErrorArray.shape

    # P_k-1
    processCovariance_Matrix = np.empty([processErrorArray_Size[0], processErrorArray_Size[0]])

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

    print(processCovariance_Matrix)

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
    firstTerm[0, 1] = 0
    firstTerm[1, 0] = 0

    # If we assume no error in our prediction process, we don't add to the firstTerm matrix
    if (qError != -1):

        finalMatrix = firstTerm + Q

        return finalMatrix
    
    else:

        return firstTerm

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

    R_Matrix = np.empty([observationErrorArray_Size[0], observationErrorArray_Size[0]])

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
dataset = DG.observedValues

#print(dataset)
# kinematicsKalman(1, 2, dataset[0][0], dataset[0][1], 20, 5, 25, 6, dataset[1][0], dataset[1][1], dataset, -1)

createKalmanFilter(1, 2, 2, dataset, obsErr, proErr)
print(dataset)