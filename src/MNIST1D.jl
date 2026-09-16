# Independent Julia implementation of https://github.com/greydanus/mnist1d.
# Dataset and templates by Sam Greydanus and Dmitry Kobak; the upstream
# transformation implementation also credits Peter Steinbach.

"""A small, configurable Julia implementation of the MNIST-1D dataset.

MNIST-1D was introduced by Greydanus and Kobak as a low-compute benchmark
for studying inductive biases in machine learning models.
"""
module MNIST1D

using Random
using Statistics

export DatasetOptions, Dataset, make_dataset, get_templates,
       trainset, testset, dataset_options, dataset, mnist1d

Base.@kwdef struct DatasetOptions
    num_samples::Int = 5_000
    train_split::Float64 = 0.8
    template_len::Int = 12
    padding::Tuple{Int, Int} = (36, 60)
    scale_coeff::Float64 = 0.4
    max_translation::Int = 48
    corr_noise_scale::Float64 = 0.25
    iid_noise_scale::Float64 = 2e-2
    shear_scale::Float64 = 0.75
    shuffle_seq::Bool = false
    final_seq_length::Int = 40
    seed::Int = 42
end

struct Dataset{T<:AbstractFloat, V<:Integer}
    x::Matrix{T}                 # observations × sequence length
    x_test::Matrix{T}
    y::Vector{V}
    y_test::Vector{V}
    t::Vector{T}
    templates::Matrix{T}
    options::DatasetOptions
end

trainset(d::Dataset) = (d.x, d.y)
testset(d::Dataset) = (d.x_test, d.y_test)
dataset_options(d::Dataset) = d.options

function Base.getproperty(d::Dataset, name::Symbol)
    name === :train && return (x=getfield(d, :x), y=getfield(d, :y))
    name === :test && return (x=getfield(d, :x_test), y=getfield(d, :y_test))
    getfield(d, name)
end

Base.propertynames(::Dataset, private::Bool=false) =
    (:train, :test, :x, :x_test, :y, :y_test, :t, :templates, :options)

function get_templates(T::Type{<:AbstractFloat}=Float64)
    digits = T[
        5 6 6.5 6.75 7 7 7 7 6.75 6.5 6 5;
        5 3 3 3.4 3.8 4.2 4.6 5 5.4 5.8 5 5;
        5 6 6.5 6.5 6 5.25 4.75 4 3.5 3.5 4 5;
        5 6 6.5 6.5 6 5 5 6 6.5 6.5 6 5;
        5 4.4 3.8 3.2 2.6 2.6 5 5 5 5 5 5;
        5 3 3 3 3 5 6 6.5 6.5 6 4.5 5;
        5 4 3.5 3.25 3 3 3 3 3.25 3.5 4 5;
        5 7 7 6.6 6.2 5.8 5.4 5 4.6 4.2 5 5;
        5 4 3.5 3.5 4 5 5 4 3.5 3.5 4 5;
        5 4 3.5 3.5 4 5 5 5 5 4.7 4.3 5
    ]
    for i in axes(digits, 1)
        row = view(digits, i, :)
        row .-= mean(row)
        row ./= std(row; corrected=false)
        row .-= row[1]
        row ./= 6
    end
    digits
end

_interp(x, n) = [begin
    p = (i - 1) * (length(x) - 1) / (n - 1) + 1
    lo = clamp(floor(Int, p), 1, length(x)); hi = clamp(ceil(Int, p), 1, length(x))
    x[lo] + (p - lo) * (x[hi] - x[lo])
end for i in 1:n]

function _gaussian_noise(rng, n, scale)
    raw = randn(rng, n) .* scale
    # Match scipy.ndimage.gaussian_filter's default sigma=2 and reflect boundary.
    kernel = [exp(-j^2 / 8) for j in -8:8]; kernel ./= sum(kernel)
    reflect(i) = (j = mod(i - 1, 2n); j < n ? j + 1 : 2n - j)
    [sum(kernel[k + 9] * raw[reflect(i + k)] for k in -8:8) for i in 1:n]
end

function _transform(rng, x, t, o)
    p = rand(rng, o.padding[1]:o.padding[2])
    xx = vcat(x .+ eps(eltype(x)), zeros(eltype(x), p))
    yy = vcat(t, zeros(eltype(t), p))
    n = o.template_len + o.padding[2]
    xx = _interp(xx, n); yy = _interp(yy, n)
    xx .*= 1 + o.scale_coeff * (rand(rng) - 0.5)
    k = o.max_translation == 0 ? 0 : rand(rng, 0:o.max_translation-1)
    k > 0 && (xx = vcat(xx[end-k+1:end], xx[1:end-k]))
    mask = xx .!= 0
    noise = _gaussian_noise(rng, length(xx), o.corr_noise_scale)
    xx = ifelse.(mask, xx, noise) .+ randn(rng, length(xx)) .* o.iid_noise_scale
    coeff = o.shear_scale * (rand(rng) - 0.5)
    xx .-= coeff .* collect(range(-0.5, 0.5; length=length(xx)))
    (_interp(xx, o.final_seq_length), _interp(yy, o.final_seq_length))
end

"""Generate a reproducible MNIST-1D dataset with train/test fields."""
function make_dataset(o::DatasetOptions=DatasetOptions(); template=get_templates(), rng=MersenneTwister(o.seed))
    o.num_samples > 0 || throw(ArgumentError("num_samples must be positive"))
    o.final_seq_length > 1 || throw(ArgumentError("final_seq_length must exceed one"))
    o.padding[1] <= o.padding[2] || throw(ArgumentError("padding must be (low, high)"))
    0 < o.train_split < 1 || throw(ArgumentError("train_split must be between zero and one"))
    o.max_translation >= 0 || throw(ArgumentError("max_translation must be nonnegative"))
    o.template_len > 1 || throw(ArgumentError("template_len must exceed one"))
    o.padding[1] >= 0 || throw(ArgumentError("padding must be nonnegative"))
    template = Matrix{Float64}(template)
    classes = size(template, 1); per_class = o.num_samples ÷ classes
    per_class > 0 || throw(ArgumentError("num_samples must be at least the number of classes"))
    o.max_translation < o.template_len + o.padding[2] ||
        throw(ArgumentError("max_translation must be shorter than the transformed sequence"))
    n = per_class * classes
    xs = Matrix{Float64}(undef, n, o.final_seq_length); ys = Vector{Int}(undef, n)
    t = collect(range(-5 / 6, 5 / 6; length=size(template, 2)))
    row = 1
    transformed_t = Vector{Float64}(undef, o.final_seq_length)
    for label in 1:classes, _ in 1:per_class
        xs[row, :], transformed_t = _transform(rng, collect(template[label, :]), t, o)
        ys[row] = label - 1; row += 1
    end
    perm = randperm(rng, n); xs = xs[perm, :]; ys = ys[perm]
    if o.shuffle_seq
        xs = xs[:, randperm(rng, o.final_seq_length)]
    end
    scale = std(xs; corrected=false)
    transformed_t ./= scale
    xs = (xs .- mean(xs)) ./ scale
    split = clamp(floor(Int, n * o.train_split), 1, n - 1)
    Dataset(xs[1:split, :], xs[split+1:end, :], ys[1:split], ys[split+1:end],
            transformed_t, Matrix(template), o)
end

"""Generate an MNIST-1D dataset using keyword options."""
dataset(; kwargs...) = make_dataset(DatasetOptions(; kwargs...))
mnist1d(; kwargs...) = dataset(; kwargs...)

end
