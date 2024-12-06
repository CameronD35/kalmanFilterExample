import numpy as np

def createKalmanFilter():
    'hi'

def kinematicsKalman(deltaT, acceleration, measuredPosition, measuredVelocity, processPositionError, processVelocityError, observationPositonError,
                     observationVelocityError, secondMeasuredPosition, secondMeasuredVelocity, dataset, prevProcessCVMatrix):

    
    #print(len(dataset))

    # A
    A_Matrix = np.matrix([
        [1, deltaT],
        [0, 1]
    ])

    # H
    H_Matrix = np.matrix([
        [1, 0],
        [0, 1]
    ])

    # X_kp
    predictedStateMatrix = getPredStateMatrix(deltaT, acceleration, measuredPosition, measuredVelocity, A_Matrix, -1)

    # P_k-1
    processCovarianceMatrix = 0

    # If we already have a previous process covariance matrix, we should not create an initial one
    if (type(prevProcessCVMatrix) is int):
        processCovarianceMatrix = getInitProcessCVMatrix(processPositionError, processVelocityError)
    else:
        #print(prevProcessCVMatrix)
        processCovarianceMatrix = prevProcessCVMatrix

    # P_kp
    predictedProcessCovarianceMatrix = getPredProcessCVMatrix(processCovarianceMatrix, A_Matrix, -1)

    # K
    kalminGain = getKalmanGain(predictedProcessCovarianceMatrix, observationPositonError, observationVelocityError, H_Matrix)

    # Y_k
    newObservation = getNewObservation(secondMeasuredPosition, secondMeasuredVelocity, -1)

    # X_k
    filteredState = calculateFilteredState(kalminGain, predictedStateMatrix, newObservation, H_Matrix)

    # P_k
    updatedProcessCovarianceMatrix = updateProcessCVMatrix(kalminGain, H_Matrix, predictedProcessCovarianceMatrix)

    #print(updatedProcessCovarianceMatrix)

    print(f'The predicted position is: {filteredState[0, 0]}.\nThe predicted velocity is: {filteredState[1, 0]}\nThe new process covariance matrix is:\n{updatedProcessCovarianceMatrix}')

    # Removes the very first item from the dataset tuple to set up for recursion
    slicedDataset = dataset[1:len(dataset)]
    #print(slicedDataset)

    # Base case: Since the dataset would not have an index of 1 (see that index 0 and 1 are necessary for the filter), the function is exited 
    if (len(slicedDataset) <= 1):
        print('Kalmin filter finished')
        return
    
    # RECURSION!
    return kinematicsKalman(deltaT, acceleration, filteredState[0, 0], filteredState[1, 0], 20, 5, 25, 6, slicedDataset[1][0], slicedDataset[1][1], slicedDataset, updatedProcessCovarianceMatrix)


# Predicted State Matrix
def getPredStateMatrix(deltaT, acceleration, measuredPosition, measuredVelocity, A, error_Matrix):
    #print('getting predicted state matrix')
    
    # A
    A_Matrix = A

    #print(A_Matrix)

    # X_k-1
    previousState_Matrix = np.matrix([
        [measuredPosition],
        [measuredVelocity]
    ])

    # B
    B_Matrix = np.matrix([
        [(0.5)*(deltaT**2)],
        [deltaT]
    ])

    # mu_k
    controlVariable_Matrix = np.matrix([
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
def getInitProcessCVMatrix(processPositionError, processVelocityError):
    #print('getting process covariance matrix')

    # P_k-1
    processCovariance_Matrix = np.matrix([
        [processPositionError**2, processPositionError*processVelocityError],
        [processPositionError*processVelocityError, processVelocityError**2]
    ])

    # Corrects the off-diagonal number. This indicates that our error in velocity does not effect the error in position and vice versa. (Done for simplicity, may not reflect reality)
    processCovariance_Matrix[0, 1] = 0
    processCovariance_Matrix[1, 0] = 0

    #print(processCovariance_Matrix)

    return processCovariance_Matrix

# Predicted Process Covariance Matrix
def getPredProcessCVMatrix(initProcessCVMatrix, A, Q):
    
    #A
    A_Matrix = A

    # A * P_k-1 * transpose(A)
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

def getKalmanGain(predProcessCVMatrix, observationPositionError, observationVelocityError, H):
    
    # H
    H_Matrix = H

    # transpose(H)
    H_Transposed = np.transpose(H_Matrix)

    # P_kp * transpose(H)
    numerator = np.dot(predProcessCVMatrix, H_Transposed)

    # H * P_kp * H^T
    denominatorFirstTerm = np.dot(np.dot(H_Matrix, predProcessCVMatrix), H_Transposed)

    # R, here we introduce the errors (either estimated or given by datasheets, us, etc.)
    R_Matrix = [
        [observationPositionError**2, 1],
        [1, observationVelocityError**2]
    ]

    # denominatorFirstTerm + R
    denominator = np.add(denominatorFirstTerm, R_Matrix)

    # numerator/denominator
    kalminGain = np.divide(numerator, denominator)

    return kalminGain


def getNewObservation(measuredPosition, measuredVelocity, error_Matrix):
    
    # C
    C_Matrix = np.matrix([
        [1, 0],
        [0, 1]
    ])

    # Y_km
    measuredValues_Matrix = np.matrix([
        [measuredPosition],
        [measuredVelocity]
    ])

    # C * Y_km
    reformattedY_Matrix = np.dot(C_Matrix, measuredValues_Matrix)

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

    # I
    identity_Matrix = np.matrix([
        [1, 0],
        [0, 1]
    ])

    # K * H
    reformattedKalminGain_Matrix = np.dot(kalminGain, H_Matrix)

    # I - above matrix
    processCVFactor_Matrix = np.subtract(identity_Matrix, reformattedKalminGain_Matrix)

    # Above Matrix * P_kp
    updatedProcessCV_Matrix = np.dot(processCVFactor_Matrix, predProcessCV_Matrix)

    return updatedProcessCV_Matrix

# (Position, Velocity)
dataset = ((4000, 280), (4260, 282), (4550, 285), (4860, 286), (5110, 290))

kinematicsKalman(1, 2, dataset[0][0], dataset[0][1], 20, 5, 25, 6, dataset[1][0], dataset[1][1], dataset, -1)