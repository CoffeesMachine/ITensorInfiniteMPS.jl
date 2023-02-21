# TODO: Move to ITensors.jl
function setval(qnval::ITensors.QNVal, val::Int)
  return ITensors.QNVal(ITensors.name(qnval), val, ITensors.modulus(qnval))
end

# TODO: Move to ITensors.jl
function Base.:/(qnval::ITensors.QNVal, n::Int)
  div_val = ITensors.val(qnval) / n
  if !isinteger(div_val)
    error("Dividing $qnval by $n, the resulting QN value is not an integer")
  end
  return setval(qnval, Int(div_val))
end

function Base.:*(qnval::ITensors.QNVal, n::Int)
  div_val = ITensors.val(qnval) * n
  if !isinteger(div_val)
    error("Multiplying $qnval by $n, the resulting QN value is not an integer")
  end
  return setval(qnval, Int(div_val))
end

# TODO: Move to ITensors.jl
function Base.:/(qn::QN, n::Int)
  return QN(map(qnval -> qnval / n, qn.data))
end

function Base.:*(qn::QN, n::Int)
  return QN(map(qnval -> qnval * n, qn.data))
end

# of Index (Tuple, Vector, ITensor, etc.)
indtype(i::Index) = typeof(i)
indtype(T::Type{<:Index}) = T
indtype(is::Tuple{Vararg{<:Index}}) = eltype(is)
indtype(is::Vector{<:Index}) = eltype(is)
indtype(A::ITensor...) = indtype(inds.(A))

indtype(tn1, tn2) = promote_type(indtype(tn1), indtype(tn2))
indtype(tn) = mapreduce(indtype, promote_type, tn)

function infsiteinds(s::Vector{<:Index}, translator=translatecelltags)
  return CelledVector(addtags(s, celltags(1)), translator)
end

shift_flux_to_zero(s::Vector{Index{Int}}, initestate::Function) = s
shift_flux_to_zero(s::Vector{Index{Int}}, flux_density::QN) = s

function shift_flux_to_zero(s::Vector{<:Index}, initstate::Function)
  return shift_flux_to_zero(s, flux(MPS(s, initstate)))
end

function shift_flux(qnblock::Pair{QN,Int}, flux_density::QN)
  return ((ITensors.qn(qnblock) - flux_density) => ITensors.blockdim(qnblock))
end
function shift_flux(space::Vector{Pair{QN,Int}}, flux_density::QN)
  return map(qnblock -> shift_flux(qnblock, flux_density), space)
end
function shift_flux(i::Index, flux_density::QN)
  return ITensors.setspace(i, shift_flux(space(i), flux_density))
end

function multiply_flux(qnblock::Pair{QN,Int}, flux_factor::Int64)
  return ((ITensors.qn(qnblock) * flux_factor) => ITensors.blockdim(qnblock))
end
function multiply_flux(space::Vector{Pair{QN,Int}}, flux_factor::Int64)
  return map(qnblock -> multiply_flux(qnblock, flux_factor), space)
end
function multiply_flux(i::Index, flux_factor::Int64)
  return ITensors.setspace(i, multiply_flux(space(i), flux_factor))
end

function shift_flux_to_zero(s::Vector{<:Index}, flux::QN)
  if iszero(flux)
    return s
  end
  n = length(s)
  try
    flux_density = flux / n
    return map(sₙ -> shift_flux(sₙ, flux_density), s)
  catch e
    s = map(sₙ -> multiply_flux(sₙ, n), s)
    return map(sₙ -> shift_flux(sₙ, flux), s)
  end
end

function infsiteinds(
  site_tag, n::Int; translator=translatecelltags, initstate=nothing, kwargs...
)
  s = siteinds(site_tag, n; kwargs...)
  s = shift_flux_to_zero(s, initstate)
  return infsiteinds(s, translator)
end

function ITensors.linkinds(ψ::InfiniteMPS)
  N = nsites(ψ)
  return CelledVector([linkinds(ψ, (n, n + 1)) for n in 1:N], translator(ψ))
end

function InfMPS(s::Vector, f::Function, translator::Function=translatecelltags)
  return InfMPS(infsiteinds(s, translator), f)
end

function indval(iv::Pair)
  return ind(iv) => val(iv)
end

zero_qn(i::Index{Int}) = nothing

function zero_qn(i::Index)
  return zero(qn(first(space(i))))
end

function insert_linkinds!(A; left_dir=ITensors.Out)
  # TODO: use `celllength` here
  N = nsites(A)
  l = CelledVector{indtype(A)}(undef, N, translator(A))
  n = N
  s = siteind(A, 1)
  dim = if hasqns(s)
    kwargs = (; dir=left_dir)
    qn_ln = zero_qn(s)
    [qn_ln => 1] #Default to 0 on the right
  else
    kwargs = ()
    1
  end
  l[N] = Index(dim, default_link_tags("l", n, 1); kwargs...)
  for n in 1:(N - 1)
    # TODO: is this correct?
    dim = if hasqns(s)
      qn_ln = flux(A[n]) * left_dir + qn_ln#Fixed a bug on flux conservation
      [qn_ln => 1]
    else
      1
    end
    l[n] = Index(dim, default_link_tags("l", n, 1); kwargs...)
  end
  for n in 1:N
    A[n] = A[n] * onehot(l[n - 1] => 1) * onehot(dag(l[n]) => 1)
  end

  @assert all(i -> flux(i) == zero_qn(s), A) "Flux not invariant under one unit cell translation, not implemented"

  return A
end

function UniformMPS(s::CelledVector, f::Function; left_dir=ITensors.Out)
  sᶜ¹ = s[Cell(1)]
  A = InfiniteMPS([ITensor(sⁿ) for sⁿ in sᶜ¹], translator(s))
  #A.data.translator = translator(s)
  N = length(sᶜ¹)
  for n in 1:N
    Aⁿ = A[n]
    Aⁿ[indval(s[n] => f(n))] = 1.0
    A[n] = Aⁿ
  end
  insert_linkinds!(A; left_dir=left_dir)
  return A
end

function InfMPS(s::CelledVector, f::Function)
  # TODO: rename cell_length
  N = length(s)
  ψL = UniformMPS(s, f; left_dir=ITensors.Out)
  ψR = UniformMPS(s, f; left_dir=ITensors.In)
  ψC = InfiniteMPS(N, translator(s))
  l = linkinds(ψL)
  r = linkinds(ψR)
  for n in 1:N
    ψCₙ = ITensor(dag(l[n])..., r[n]...)
    ψCₙ[l[n]... => 1, r[n]... => 1] = 1.0
    ψC[n] = ψCₙ
  end
  return ψ = InfiniteCanonicalMPS(ψL, ψC, ψR)
end

function ITensors.expect(ψ::InfiniteCanonicalMPS, o::String, n::Int)
  s = siteinds(only, ψ.AL)
  O = op(o, s[n])
  ϕ = ψ.AL[n] * ψ.C[n]
  return inner(ϕ, apply(O, ϕ))
end

function ITensors.expect(ψ::InfiniteCanonicalMPS, h::MPO)
  l = linkinds(ITensorInfiniteMPS.only, ψ.AL)
  r = linkinds(ITensorInfiniteMPS.only, ψ.AR)
  s = siteinds(ITensorInfiniteMPS.only, ψ)
  δˢ(n) = ITensorInfiniteMPS.δ(dag(s[n]), prime(s[n]))
  δˡ(n) = ITensorInfiniteMPS.δ(l[n], prime(dag(l[n])))
  δʳ(n) = ITensorInfiniteMPS.δ(dag(r[n]), prime(r[n]))
  ψ′ = prime(dag(ψ))

  ns = ITensorInfiniteMPS.findsites(ψ, h)
  nrange = ns[end] - ns[1] + 1
  idx = 2
  temp_O = δˡ(ns[1] - 1) * ψ.AL[ns[1]] * h[1] * ψ′.AL[ns[1]]
  for n in (ns[1] + 1):(ns[1] + nrange - 1)
    if n == ns[idx]
      temp_O = temp_O * ψ.AL[n] * h[idx] * ψ′.AL[n]
      idx += 1
    else
      temp_O = temp_O * (ψ.AL[n] * δˢ(n)) * ψ′.AL[n]
    end
  end
  temp_O = temp_O * (ψ.C[ns[end]] * denseblocks(δʳ(ns[end])) * ψ′.C[ns[end]])
  return temp_O[]
end

function ITensors.expect(ψ::InfiniteCanonicalMPS, h::InfiniteSum)
  return [expect(ψ, h[j]) for j in 1:nsites(ψ)]
end

function ent_spec(psi::InfiniteCanonicalMPS)
  n = nsites(psi)
  ent_spec = []
  for x in 1:n
    localC = psi.C[x]
    if isdiag(localC)
      println("Blah")
      append!(ent_spec, [localC])
    else
      linkl = only(commoninds(psi.C[x], psi.AL[x]))
      U, S, V = svd(localC, [linkl])
      append!(ent_spec, [S])
    end
  end
  return ent_spec
end

function entropies(psi::InfiniteCanonicalMPS)
  n = nsites(psi)
  entropies = zeros(n)
  for x in 1:n
    ent = 0
    localC = psi.C[x]
    if isdiag(localC)
      for s in 1:size(localC)[1]
        ent += -2 * localC[s, s]^2 * log(localC[s, s])
      end
    else
      linkl = only(commoninds(psi.C[x], psi.AL[x]))
      U, S, V = svd(localC, [linkl])
      for s in 1:size(S)[1]
        ent += -2 * S[s, s]^2 * log(S[s, s])
      end
    end
    entropies[x] = ent
  end
  return entropies
end


function ITensors.truncate!(psi::InfiniteCanonicalMPS; kwargs...)
  n = nsites(psi)
  site_range=get(kwargs, :site_range, 1:n+1)

  s = siteinds(only, psi.AL)
  for j in first(site_range):last(site_range)-1
    left_indices = [ only(filter(x->dir(x) == ITensors.Out, commoninds(psi.AL[j], psi.AL[j-1]))), s[j] ]
    new_tag = tags(only(commoninds(psi.AL[j], psi.C[j])))
    U, S, V = svd(psi.AL[j]*psi.C[j]*psi.AR[j+1], left_indices, lefttags=new_tag, righttags = new_tag; kwargs...)
    psi.AL[j] = U
    psi.AR[j+1] = V
    psi.C[j] = denseblocks(itensor(S))
    #TODO this choice preserve the AL C = C AR on the untouched bonds, but not on the middle. Is it really the best choice?
    # Currently, in the iDMRG, I also update the Cleft (the Cright is updated next)
    # Note that it in principle does not really matter when doing the iDMRG
    temp_R = ortho_polar(U * S, psi.C[j - 1])
    psi.AR[j] = temp_R
    temp_L = ortho_polar(S * V, psi.C[j + 1])
    psi.AL[j+1] = temp_L
  end
end
