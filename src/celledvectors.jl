
struct Cell{T}
  cell::T
end

#
# cells
#
celltagprefix() = "c="
celltags(n::Integer) = TagSet(celltagprefix() * string(n))
celltags(n1::Integer, n2::Integer) = TagSet(celltagprefix() * n1 * "|" * n2)

indextagprefix() = "n="
#
# translatecell
#
# TODO: account for shifting by a tuple, for example:
# translatecell(ts"Site,c=1|2", (2, 3)) -> ts"Site,c=3|5"
# TODO: ts"c=10|12" has too many characters
# TODO: ts"c=1|2|3" has too many characters
#

# Determine the cell `n` from the tag `"c=n"`
function getcell(ts::TagSet)
  celltag = tag_starting_with(ts, celltagprefix())
  if isnothing(celltag) #dealing with link legs
    return celltag
  end
  return parse(Int, celltag[(length(celltagprefix()) + 1):end])
end

function getsite(ts::TagSet)
  celltag = tag_starting_with(ts, indextagprefix())
  return parse(Int, celltag[(length(indextagprefix()) + 1):end])
end


##Translation operators

#Default translate cell
function translatecell(ts::TagSet, n::Integer)
  ncell = getcell(ts)
  if isnothing(ncell)
    return ts
  end
  return replacetags(ts, celltags(ncell) => celltags(ncell + n))
end

function translatecell(i::Index, n::Integer)
  ts = tags(i)
  translated_ts = translatecell(ts, n)
  return replacetags(i, ts => translated_ts)
end

#Transfer the functional properties
#translatecell(translater, T::ITensor, n::Integer) = translater(T, n)
translatecell(translater::Function, T::ITensor, n::Integer) = ITensors.setinds(T, translatecell(translater, inds(T), n))
translatecell(translater::Function, T::MPO, n::Integer) = translatecell.(translater::F, T, n)
translatecell(translater::Function, T::Matrix{ITensor}, n::Integer) = translatecell.(translater::F, T, n)
translatecell(translater::Function, i::Index, n::Integer) = translater(i, n)
function translatecell(translater, is::Union{<:Tuple,<:Vector}, n::Integer)
  return translatecell.(translater, is, n)
end

#Default behavior
#translatecell(T::ITensor, n::Integer) = ITensors.setinds(T, translatecell(inds(T), n))
#translatecell(T::MPO, n::Integer) = translatecell.(T, n)
#translatecell(T::Matrix{ITensor}, n::Integer) = translatecell.(T, n)


## CelledVector definition
struct CelledVector{T, F} <: AbstractVector{T}
  data::Vector{T}
  translater::F
end

ITensors.data(cv::CelledVector) = cv.data
Base.convert(::Type{CelledVector{T}}, v::Vector) where {T} = CelledVector{T}(v)
translater(cv::CelledVector) = cv.translater

function CelledVector{T}(::UndefInitializer, n::Integer) where {T}
  return CelledVector(Vector{T}(undef, n))
end

function CelledVector{T}(::UndefInitializer, n::Integer, translater::Function) where {T}
  return CelledVector(Vector{T}(undef, n), translater::Function)
end

CelledVector(v::AbstractVector) = CelledVector(v, mytranslatecell)


function mytranslatecell(i::Index, n::Integer)
  println("Blah")
  return translatecell(i::Index, n::Integer)
end
"""
    celllength(cv::CelledVector)

The length of a unit cell of a CelledVector.
"""
celllength(cv::CelledVector) = length(ITensors.data(cv))

# For compatibility with Base
Base.size(cv::CelledVector) = size(ITensors.data(cv))

"""
    cell(cv::CelledVector, n::Integer)

Which unit cell index `n` is in.
"""
function cell(cv::CelledVector, n::Integer)
  _cell = fld1(n, celllength(cv))
  return _cell
end

"""
    cellindex(cv::CelledVector, n::Integer)

Which index in the unit cell index `n` is in.
"""
cellindex(cv::CelledVector, n::Integer) = mod1(n, celllength(cv))

# Get the value at index `n`, where `n` must
# be within the first unit cell
_getindex_cell1(cv::CelledVector, n::Int) = ITensors.data(cv)[n]

# Set the value at index `n`, where `n` must
# be within the first unit cell
_setindex_cell1!(cv::CelledVector, val, n::Int) = (ITensors.data(cv)[n] = val)

# Fallback
#translatecell(x, ::Integer) = x # I think this is useless now

function getindex(cv::CelledVector, n::Int)
  cellₙ = cell(cv, n)
  siteₙ = cellindex(cv, n)
  return translatecell(cv.translater, _getindex_cell1(cv, siteₙ), cellₙ - 1)
end

# Do we need this definition? Maybe uses generic Julia fallback
#getindex(cv::CelledVector, r::AbstractRange) = [cv[n] for n in r]

function Base.firstindex(cv::CelledVector, c::Cell)
  return (c.cell - 1) * celllength(cv) + 1
end

function Base.lastindex(cv::CelledVector, c::Cell)
  return c.cell * celllength(cv)
end

function Base.keys(cv::CelledVector, c::Cell)
  return firstindex(cv, c):lastindex(cv, c)
end

## function Base.eachindex(cv::CelledVector, c::Cell)
##   return firstindex(cv, c):lastindex(cv, c)
## end

getindex(cv::CelledVector, c::Cell) = cv[eachindex(cv, c)]

function setindex!(cv::CelledVector, T, n::Int)
  cellₙ = cell(cv, n)
  siteₙ = cellindex(cv, n)
  _setindex_cell1!(cv, translatecell(cv.translater, T, -(cellₙ - 1)), siteₙ)
  return cv
end

celltags(cell) = TagSet("c=$cell")
