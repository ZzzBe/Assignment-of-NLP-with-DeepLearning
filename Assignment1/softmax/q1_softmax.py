import numpy as np
from sklearn.utils.extmath import softmax

def Jacksoftmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape
    x.astype('float64')
    #print x
    #x = np.exp(x)
    #print x
    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        max_row = np.max(x, axis=1)
        e_x = np.exp(x - max_row[:,np.newaxis])
        div = np.sum(e_x, axis=1)
        x = e_x/div[:,np.newaxis]
        #print x
        #raise NotImplementedError
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        x = np.exp((x-x.min())/(x.max()-x.min()))
        sum_column = sum(x)
        for i in range(x.size):
            x[i] = x[i]/float(sum_column)
        #raise NotImplementedError
        ### END YOUR CODE
    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = Jacksoftmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = Jacksoftmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = Jacksoftmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    ##
    print "Enter the dimension of the matrix:"
    n = input()
    m = input()
    test1 = np.random.random((n,m))
    print test1
    sk_array1 = softmax(test1)
    print sk_array1
    jk_array1 = Jacksoftmax(test1)
    print jk_array1
    assert np.allclose(sk_array1, jk_array1, rtol=1e-05, atol=1e-06)

    test2 = 2.5 * np.random.randn(n,m)+3
    print test2
    sk_array2 = softmax(test2)
    print sk_array2
    jk_array2 = Jacksoftmax(test2)
    print jk_array2
    assert np.allclose(sk_array2, jk_array2, rtol=1e-05, atol=1e-06)
    #raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
