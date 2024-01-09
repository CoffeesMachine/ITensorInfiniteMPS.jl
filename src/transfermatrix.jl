# TODO: make a TransferMatrix type?

TransferMatrix(ψ::InfiniteMPS) = TransferMatrix(ψ, Cell(1))

function TransferMatrix(ψ::InfiniteMPS, c::Cell)
  N = nsites(ψ)
  ψᴴ = prime(linkinds, dag(ψ))
  ψᶜ = ψ[c]
  ψᶜᴴ = ψᴴ[c]
  r = unioninds(linkinds(ψ, N => N + 1), linkinds(ψᴴ, N => N + 1))
  l = unioninds(linkinds(ψ, 1 => 0), linkinds(ψᴴ, 1 => 0))
  return ITensorMap(ψᶜ, ψᶜᴴ; input_inds=r, output_inds=l)
end


function TransferMatrix(ψ1::InfiniteMPS, ψ2::InfiniteMPS; direction=+1)
  c = Cell(1)
  @assert nsites(ψ1) == nsites(ψ2)

  N = nsites(ψ1)
  ψᴴ = prime(linkinds, dag(ψ1))
  ψ2ᶜ = ψ2[c]
  ψ1ᶜᴴ = ψᴴ[c]

  

  r = unioninds(linkinds(ψ2, N => N+1), linkinds(ψᴴ, N => N+1))
  l = unioninds(linkinds(ψ2, 1 => 0), linkinds(ψᴴ, 1 => 0))
  
  return ITensorNetworkInfinite(ψ2ᶜ, ψ1ᶜᴴ; input_inds=r, output_inds=l, translator=translator(ψ1), direction)
end