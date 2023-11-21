import LinearAlgebra

function pos_def_projection(M :: AbstractMatrix, eigenvalue_epsilon = 1e-6)

    evals, evecs = LinearAlgebra.eigen(Symmetric(M))
    if minimum(evals) <= eigenvalue_epsilon
        M = Symmetric(evecs * LinearAlgebra.diagm(max.(eigenvalue_epsilon, evals)) * evecs')
    end

    # U, S, V = svd(Symmetric(M))
    # if minimum(S) < eigenvalue_epsilon
    #     M = Symmetric(U * Diagonal(max.(eigenvalue_epsilon, S)) * V')
    # end

    smallest_eig = LinearAlgebra.eigmin(Symmetric(M))
    if smallest_eig <= 0.0
        M += (eigenvalue_epsilon - smallest_eig) * I
    end

    return M
end