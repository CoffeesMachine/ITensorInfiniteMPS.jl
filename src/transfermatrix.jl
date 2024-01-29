using KrylovKit
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


function TransferMatrix(ψ1::InfiniteMPS, ψ2::InfiniteMPS; i::Int64=1, direction=+1)
  @assert nsites(ψ1) == nsites(ψ2)
  cell = Cell(i)
  N = nsites(ψ1) 
  ψᴴ = prime(linkinds, dag(ψ1))
  ψ2ᶜ = ψ2[cell]
  ψ1ᶜᴴ = ψᴴ[cell]

  

  r = unioninds(linkinds(ψ2, i*N => i*N+1), linkinds(ψᴴ, i*N => i*N+1))
  l = unioninds(linkinds(ψ2, (i-1)*N + 1 => (i-1)*N), linkinds(ψᴴ, (i-1)*N + 1 => (i-1)*N))
  
  return ITensorNetworkInfinite(ψ2ᶜ, ψ1ᶜᴴ; input_inds=r, output_inds=l, translator=translator(ψ1), direction)
end




function EigenStatesTransferMatrix(ψ::InfiniteCanonicalMPS; Neigs::Int64=1, returnEigs=false)
    
  Tᴿ =  TransferMatrix(ψ.AR, ψ.AR)
  Tᴸ = transpose(TransferMatrix(ψ.AL, ψ.AL))
  vⁱᴿ = randomITensor(dag(input_inds(Tᴿ)))
  vⁱᴸ = randomITensor(dag(input_inds(Tᴸ)))
  
  vⁱᴿ = translatecell(translator(ψ), vⁱᴿ, -1)
  vⁱᴸ = translatecell(translator(ψ), vⁱᴸ, 1)


  λᴿ, vᴿ, _ = KrylovKit.eigsolve(Tᴿ, vⁱᴿ, Neigs, :LM; tol=1e-10)
  λᴸ, vᴸ, _ = KrylovKit.eigsolve(Tᴸ, vⁱᴸ, Neigs, :LM; tol=1e-10)
 
  translateLeft(v::ITensor) = translatecell(translator(ψ), v, -1)
  translateRight(v::ITensor) = translatecell(translator(ψ), v, 1)

  vᴿ = translateRight.(vᴿ)
  vᴸ = translateLeft.(vᴸ)



  if returnEigs
      return (λᴸ[1:Neigs], vᴸ[1:Neigs]), (λᴿ[1:Neigs], vᴿ[1:Neigs])
  else
      return vᴸ[1:Neigs] , vᴿ[1:Neigs]
  end
end

function LeftNormalizedEigenstates(ψ::InfiniteMPS; Neigs::Int64=1, returnEigs=false, which=:L)
  if which==:L
    Tᴸ = transpose(TransferMatrix(ψ, ψ))
    vⁱᴸ = randomITensor(dag(input_inds(Tᴸ)))
    vⁱᴸ = translatecell(translator(ψ), vⁱᴸ, 1)
    λᴸ, vᴸ, _ = KrylovKit.eigsolve(Tᴸ, vⁱᴸ, Neigs, :LM; tol=1e-10)
  
    translateLeft(v::ITensor) = translatecell(translator(ψ), v, -1)

    
    vᴸ = translateLeft.(vᴸ)
    if returnEigs
        return λᴸ[1:Neigs], vᴸ[1:Neigs]
    else
        return vᴸ[1:Neigs] 
    end




  elseif which==:R
    Tᴿ = TransferMatrix(ψ, ψ)
    vⁱᴿ = randomITensor(dag(input_inds(Tᴿ)))
    vⁱᴿ = translatecell(translator(ψ), vⁱᴿ, -1)
    λᴿ, vᴿ, _ = KrylovKit.eigsolve(Tᴿ, vⁱᴿ, Neigs, :LM; tol=1e-10)
  
    translateRight(v::ITensor) = translatecell(translator(ψ), v, 1)

    vᴿ = translateRight.(vᴿ)
    if returnEigs
        return λᴿ[1:Neigs], vᴿ[1:Neigs]
    else
        return vᴿ[1:Neigs]
    end
  end
end

###########################################
#                                         #
# Expection value and correlation matrix  #
#                                         #
###########################################


function ITensors.expect(ψ::InfiniteCanonicalMPS, NameOp::AbstractString; sites::AbstractRange=1:nsites(ψ), L=nothing, R=nothing)
    
  
  if isnothing(L) || isnothing(R)
      L, R = EigenStatesTransferMatrix(copy(ψ); Neigs=1, returnEigs=false)
  end
  
  UnitCell = expectationUnitCell(ψ, NameOp, L[1], R[1])

  K = repeat(UnitCell, div(length(sites), nsites(ψ))) 
  
  return K

end

function expectationUnitCell(ψⁱ::InfiniteCanonicalMPS, NameOp::AbstractString, L::ITensor, R::ITensor)
  
  
  UnitCell = Array{Any}(undef, nsites(ψⁱ)) 
  s = [inds(ψⁱ[j])[2] for j=1:nsites(ψⁱ)]
  
  baseR = (ψⁱ.C[nsites(ψⁱ)]*R)*dag(ψⁱ.C[nsites(ψⁱ)])'
  normpsi2 = (Contract(copy(ψⁱ.AL), L)*baseR)[]
  for i=1:nsites(ψⁱ)
      ψ = copy(ψⁱ.AL[Cell(1)])
      baseL = L 
      pL = 1 
      while pL < i 
        baseL = (baseL * ψ[pL])*prime(dag(ψ[pL]), !s[pL])
        pL += 1
      end 
      Li = baseL * ψ[i]

      o = op(NameOp, s[i])

      baseL = (o*Li)*dag(ψ[i])'

      while pL < nsites(ψⁱ)
        pL += 1
        baseL = (baseL * ψ[pL])*prime(dag(ψ[pL]), !s[pL])
      end 

      Ov = (baseL*baseR)[]
      
      UnitCell[i] = Ov/normpsi2
  end
  return UnitCell
end


function ITensors.correlation_matrix(psi::InfiniteCanonicalMPS, OP1::AbstractString, OP2::AbstractString; sites::AbstractRange=1:nsites(psi))
  Ns = nsites(psi)
  
  
  λ, LeftEigenSet = LeftNormalizedEigenstates(copy(psi.AL); Neigs=2, returnEigs=true, which=:L)
  RightEigenSet = LeftNormalizedEigenstates(copy(psi.AR); Neigs=2, returnEigs=false, which=:R)

  UnitCell(n) = n > 0 ? div(n-1, Ns) : div(n, Ns)-1
  Exp = expectationUnitCell(psi, OP1, LeftEigenSet[1], RightEigenSet[1])
  C= correlationElement(psi, OP1, OP2, LeftEigenSet, RightEigenSet, λ[2], Exp; sites=sites)
  
  CorrelationMatrix = zeros(length(sites), length(sites))
  
  for i in eachindex(sites)
    Cell_i = UnitCell(i)+1
    pos_i = mod(i-1, Ns)+1
    for j in eachindex(sites)
      j < i && continue
      Cell_j = UnitCell(j)+1
      pos_j = mod(j-1, Ns)+1 

      CorrelationMatrix[i, j] = C[pos_i, pos_j + Ns*abs(Cell_i-Cell_j)]
      CorrelationMatrix[j, i] = CorrelationMatrix[i, j]
    end
  end

  return CorrelationMatrix
end



function correlationElement(ψ::InfiniteCanonicalMPS, OP1::AbstractString, OP2::AbstractString, LeftSet, RightSet, λ, Exp; sites::AbstractRange=1:nsites(ψ))
  psi = copy(ψ.AL)
  Ns = nsites(ψ)

  normpsi2 = normPsi(ψ, LeftSet[1], RightSet[1])
  normLeft = Contract(ψ.AL, LeftSet[1]; which=:L)
  
  C = zeros(Ns, length(sites))
  lastCell = div(length(sites)-1, Ns) + 1
  L_corr = ceil(-Ns/log(abs(λ)))
  println("Estimation of correlation length : $L_corr")
  flush(stdout)

  #Calulate the siteinds 
  s = [inds(psi[i])[2] for i=1:Ns]

  for i in 1:Ns
    Cell_i = 1
    n_i =  mod(i-1, Ns) + 1

    ψL = copy(psi[Cell(1)])
    pL = 1
    baseL = LeftSet[1]
    while pL < n_i
      baseL = (baseL*ψL[pL])*prime(dag(ψL[pL]), !s[pL])
      pL += 1
    end

    Li = baseL * ψL[n_i]
    
    #case i==j
    o = op( "$OP1 * $OP2", s[n_i])

    localL = copy(baseL) 
    localL = (dag(ψL[n_i])' * o)*Li

    while pL < Ns
      pL += 1 
      localL = (localL*ψL[pL])*prime(dag(ψL[pL]), !s[pL])
    end 
    localL = (localL*ψ.C[Ns])*dag(ψ.C[Ns])'
    c = localL*RightSet[1]
    
    C[n_i, n_i] = c[1]/normpsi2 

    #case i != j 
    oᵢ = op(OP1, s[n_i])
    baseL = (dag(ψL[n_i])' * oᵢ) * Li
    
  
  
    # Case where j is in the same unit Cell as i 
    for n_j in i+1:Ns
      L_loc = copy(baseL)
      pL = n_i+1
      while pL < n_j 
        L_loc = (L_loc*ψL[pL])*prime(dag(ψL[pL]), !s[pL])
        pL += 1
      end
      Lj = L_loc * ψL[n_j]
      oᵢ = op(OP2, s[n_j])
      L_loc = (dag(ψL[n_j])' * oᵢ) * Lj

      while pL < Ns
        pL += 1 
        L_loc = (L_loc*ψL[pL])*prime(dag(ψL[pL]), !s[pL])
      end

      L_loc = (L_loc*ψ.C[Ns])*dag(ψ.C[Ns])'
      c = L_loc*RightSet[1]
      
      C[n_i, n_j] = c[1]/normpsi2
    end 

    baseL2 = copy(baseL)
    pL = n_i+1

    while pL < Ns+1 
      baseL2 = (baseL2*ψL[pL])*prime(dag(ψL[pL]), !s[pL])
      pL += 1 
    end 
    # Case where j is in another unit cell 
    for n_j in 1:Ns
      L_loc = copy(baseL2)
      nL = copy(normLeft)
      for cell in 2:lastCell
        shift = Ns*(cell-1)
        if n_j + shift > 10*L_corr
          C[n_i, n_j + shift] = Exp[n_i]*Exp[n_j]
          continue
        end


        Rleg = (translatecell(translator(ψ), RightSet[1], cell-1)*ψ.C[Ns*cell])*dag(ψ.C[Ns*cell])'
        ψR = copy(ψ.AL[Cell(cell)])
     
        
        normRight = Contract(ψ.AL, Rleg; which=:R, ind=cell)
        pR = Ns
        baseR = copy(Rleg)
        
        while pR > n_j
          sᵢ = siteind(ψ, pR+shift)
          baseR = (baseR*ψR[pR])*prime(dag(ψR[pR]), !sᵢ)
          pR -= 1 
        end 
        
        Rj = baseR * ψR[n_j]
        oᵢ = op(OP2, siteind(ψ, n_j+shift))
        baseR = (dag(ψR[n_j])'*oᵢ)*Rj 

        while pR > 1 
          pR -= 1
          sᵢ = siteind(ψ,  pR+shift)
          baseR = (baseR*ψR[pR])*prime(dag(ψR[pR]), !sᵢ)
        end
    
        CorrEl = 0 
        diffCell = abs(cell-1)

        if cell == 2 
          L2 = copy(L_loc)
          R2 = copy(baseR)
          nn = (copy(nL)*copy(normRight))[]
          s2 = L2*R2/nn
          C[n_i, n_j + shift] = s2[]
        else
          T = TransferMatrix(ψ.AL, ψ.AL; i=cell-1)
          Counter = 2  
          L_loc = L_loc*T
          nL = nL*T
          nn = (copy(nL)*copy(normRight))[]

          C[n_i, n_j + shift] = (L_loc*baseR)[]/nn
        end
      end 
    end 

  end
  return C
end

function Contract(ψᵢ::InfiniteMPS, LegSide::ITensor; which=:L, ind=1)
  N = nsites(ψᵢ)
  ψ = copy(ψᵢ[Cell(ind)])
  s = [inds(ψᵢ[i])[2] for i=1:N]
  

  if which == :L
    L = LegSide

    for i in 1:N
      L = (L*ψ[i])*prime(dag(ψ[i]), !s[i]) 
    end 

    return L
  else 
    R = LegSide

    for i in reverse(1:N)
      sᵢ= siteind(ψᵢ, i+N*(ind-1))
      R = prime(dag(ψ[i]), !sᵢ)*(ψ[i]*R)
    end 
    
    return R 
  end 
end 


function normPsi(ψᵢ::InfiniteCanonicalMPS, Lᵢ::ITensor, Rᵢ::ITensor)
  N = nsites(ψᵢ)
  ψ = copy(ψᵢ[Cell(1)])
  s = [inds(ψᵢ[i])[2] for i=1:N]


  L = Lᵢ

  for i in 1:N
    L = (L*ψ[i])*prime(dag(ψ[i]), !s[i])
  end

  L =(L*ψᵢ.C[N])*dag(ψᵢ.C[N])'
  norm2 = L*Rᵢ
  return norm2[1]
end 
