module Etude21

using PartialFunctions, Plots, Quadmath

function unzip(list::Vector{NTuple{N, T}}) where {N, T}
    return (
        ([res[i] for res in list] for i in 1:N)
    )
end

hilbert(::Type{Rational}, n::Int) = [1 // (i + j - 1) for i ∈ 1:n, j ∈ 1:n]
hilbert(T::Type{<:AbstractFloat}, n::Int) = [oneunit(T) / (i + j - 1) for i ∈ 1:n, j ∈ 1:n]
hilbert(n::Int) = hilbert(Float64, n)

div_op(x::T, y::S) where {T<:Real, S<:Real} = x // y
div_op(x::AbstractFloat, y::AbstractFloat) = x/y
div_op(x::AbstractFloat, y::Real) = x/y
div_op(x::Real, y::AbstractFloat) = x/y

gaussian_inverse(A::Matrix{T}) where T<:Real = let A′ = copy(A)
     gaussian_inverse!(similar(A), A′)
end
gaussian_inverse(A::Matrix{T}) where T<:Integer = let A′ = Rational.(A)
        gaussian_inverse!(similar(A′), A′)
end

function gaussian_inverse!(Ainv::Matrix{T}, A::Matrix{T}) where T<:Real
    n = size(Ainv, 1)
    if n != size(Ainv, 2) || size(Ainv) != size(A)
        throw(ArgumentError("Matrices must be square and of same size"))
    end

    # set x to identity matrix
    for i ∈ 1:n, j ∈ 1:n
        Ainv[i, j] = i == j ? oneunit(T) : zero(T)
    end

    for j ∈ 1:n
        i, m = 0, 0
        for k ∈ j:n
            if abs(A[k, j]) > abs(m)
                i, m = k, A[k, j]
            end
        end

        m == 0 && throw(ArgumentError("Matrix is singular"))

        for k ∈ 1:n
            A[j, k], A[i, k] = A[i, k],  A[j, k]
            A[j, k] = div_op(A[j, k], m)

            Ainv[j, k], Ainv[i, k] = Ainv[i, k], Ainv[j, k]
            Ainv[j, k] = div_op(Ainv[j, k], m)
        end

        for i ∈ 1:n
            i == j && continue
            m = A[i, j]
            for k ∈ 1:n
                if k >= j
                    A[i, k] = A[i, k] - m*A[j, k]
                end
                Ainv[i, k] = Ainv[i, k] - m*Ainv[j, k]
            end
        end
    end

    return Ainv
end

function row_norm(A)
    n = size(A, 1)
    maximum(
        sum(abs(A[i, j]) for j in 1:n) for i in 1:n
    )
end

function column_norm(A)
    n = size(A, 1)
    maximum(
        sum(abs(A[i, j]) for i in 1:n) for j in 1:n
    )
end

L1_norm(A) = sum(abs.(A))
L2_norm(A) = sqrt.(sum(A.^2))
Linf_norm(A) = maximum(abs.(A))

function hilbert_test(T::DataType, n)
    H = hilbert(T, n)
    H⁻¹ = gaussian_inverse(H)
    I = [i == j ? 1 : 0 for i ∈ 1:n, j ∈ 1:n]
    L = H⁻¹ * H - I
    R = H * H⁻¹ - I

    return row_norm(L), row_norm(R), column_norm(L), column_norm(R),
        L1_norm(L), L1_norm(R), L2_norm(L), L2_norm(R), Linf_norm(L), Linf_norm(R)
end

function perverse_inverse(max = 50, step = 5)
    n = [3;4;5:step:max]
    results_32 = unzip(map(hilbert_test $ Float32, n))
    results_64 = unzip(map(hilbert_test $ Float64, n))
    results_128 = unzip(map(hilbert_test $ Float128, n))

    bits = (32, 64, 128)
    styles = (:solid, :dash, :dashdot)
    colors = palette(:tab10)

    p = plot(; size = (600, 600), legend = :right, yaxis = ("Norm", :log), xaxis = (0:step:max, "Size of matrix", (0, 2*max)))

    foreach(zip(bits, styles, (results_32, results_64, results_128))) do (nbits, linestyle, (Lr, Rr, Lc, Rc, LL1, RL1, LL2, RL2, LLinf, RLinf))
        plot!(p, n, Lr; label = "Left row norm ($nbits bit)", linestyle, linecolor = colors[1])
        plot!(p, n, Rr; label = "Right row norm ($nbits bit)", linestyle, linecolor = colors[2])
        plot!(p, n, Lc; label = "Left column norm ($nbits bit)", linestyle, linecolor = colors[3])
        plot!(p, n, Rc; label = "Right column norm ($nbits bit)", linestyle, linecolor = colors[4])
        plot!(p, n, LL1; label = "Left L1 norm ($nbits bit)", linestyle, linecolor = colors[5])
        plot!(p, n, RL1; label = "Right L1 norm ($nbits bit)", linestyle, linecolor = colors[6])
        plot!(p, n, LL2; label = "Left L2 norm ($nbits bit)", linestyle, linecolor = colors[7])
        plot!(p, n, RL2; label = "Right L2 norm ($nbits bit)", linestyle, linecolor = colors[8])
        plot!(p, n, LLinf; label = "Left L∞ norm ($nbits bit)", linestyle, linecolor = colors[9])
        plot!(p, n, RLinf; label = "Right L∞ norm ($nbits bit)", linestyle, linecolor = colors[10])
    end

    plot!(p, n, 1e-5 *(1 + sqrt(2)).^(4.0.*n)./sqrt.(n), label = "Condition number", lw = 2)
    p
end

end