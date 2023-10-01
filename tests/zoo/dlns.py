import torch

from devinterp.zoo.dlns import DLN


def test_from_matrix():
    # Set dimensions and number of layers
    input_dim = 5
    output_dim = 3
    L = 1

    # Create a random matrix
    A = torch.randn((output_dim, input_dim))

    # Create a DLN from the matrix
    dln_from_matrix = DLN.from_matrix(A, L=L)

    # Use the DLN's to_matrix method to get the matrix representation
    A_from_dln = dln_from_matrix.to_matrix()

    # Check if the matrices are close to each other
    torch.testing.assert_allclose(A, A_from_dln, atol=1e-5, rtol=1e-5)
