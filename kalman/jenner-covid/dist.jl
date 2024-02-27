import Distributions
import LinearAlgebra

struct CustomNormal{T<:Real}
    μ::Vector{T}
    Σ::Matrix{T}
    A::Matrix{T}
    dim::Int
end

function CustomNormal(μ, Σ)
    dim = length(μ)
    evals, evecs = LinearAlgebra.eigen(Symmetric((Σ + Σ') / 2.0))
    A = evecs * LinearAlgebra.diagm(sqrt.(max.(0.0, real.(evals))))
    return CustomNormal(μ, Matrix(Σ), Matrix(A), dim)
end

function Distributions.rand(a::CustomNormal)
    return a.μ + a.A * rand(Distributions.MvNormal(LinearAlgebra.I(length(a.μ))))
end

function surprisal(a::CustomNormal, x::Vector)
    return (x - a.μ)' * (a.Σ \ (x - a.μ)) / 2.0 + logdet(a.Σ) / 2 + log(2 * pi) * a.dim / 2
end
