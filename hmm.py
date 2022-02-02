import numpy as np

def forward(A, B, pi, O):
    """
    Calculates the probability of an observation sequence O given the model(A, B, pi).
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities (N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The probability of the observation sequence and the calculated alphas in the Trellis diagram with shape
             (N, T) which should be a numpy array.
    """
    T = O.shape[0]
    N = A.shape[0]
    forward = np.zeros((N, T))
    forward[:, 0] = pi*B[:, O[0]]
    for t in range(1, T):
        for j in range(N):
            forward[j, t] = np.sum(forward[:, t - 1]*(A[:, j]) * B[j, O[t]])
    res = np.sum(forward[:, T-1])
    return res,forward



def viterbi(A, B, pi, O):
    """
    Calculates the most likely state sequence given model(A, B, pi) and observation sequence.
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities(N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The most likely state sequence with shape (T,) and the calculated deltas in the Trellis diagram with shape
             (N, T). They should be numpy arrays.
    """
    T = O.shape[0] # length of observation
    N = A.shape[0] # length of states
    viterbi = np.zeros((N, T)) # Same shape with back pointer
    back = np.zeros((N, T))
    res = np.zeros(T)
    viterbi[:, 0] = pi * B[:, O[0]] #initially
    for t in range(1, T):
        for j in range(N):
            probs = viterbi[:, t - 1] * (A[:, j]) * (B[j, O[t]])
            viterbi[j, t] = np.max(probs)
            back[j, t] = np.argmax(probs)

    best_one = np.argmax(viterbi[:, T-1]) #first element
    res[0] = best_one
    index = 1
    i = T-1
    while i > 0:
        x = int(best_one)
        res[index] = back[x, i]
        best_one = back[x, i]
        index += 1
        i-=1

    return np.flip(res),viterbi
